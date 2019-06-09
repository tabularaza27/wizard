# from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
import tf_agents
import tf_agents.agents
import tf_agents.trajectories.time_step as ts
import tf_agents.replay_buffers

from tensorflow.contrib.framework import TensorSpec, BoundedTensorSpec
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from agents.rl_agent import RLAgent, STATE_DIMENSIONS, ACTION_DIMENSIONS, MODELS_PATH
from agents.tf_agents.layers import equal_spacing_fc
from agents.tf_agents.networks import MaskedActorNetwork, DummyMaskedValueNetwork

REPLAY_BUFFER_SIZE = 10 * sum(range(1, 16)) # 10 games

def _to_tf_timestep(time_step: ts.TimeStep) -> ts.TimeStep:
    """Batch & convert all arrays to tensors in the input timestep"""

    time_step = tf_agents.utils.nest_utils.batch_nested_array(time_step)
    return tf.contrib.framework.nest.map_structure(tf.convert_to_tensor, time_step)

class TFAgentsPPOAgent(RLAgent):
    def __init__(self, name=None, actor_net=None, value_net=None,
            predictor=None, keep_models_fixed=False):
        super().__init__(name, predictor, keep_models_fixed)

        action_spec = BoundedTensorSpec((1,), tf.int64, 0, ACTION_DIMENSIONS - 1)

        # we store both mask and the actual observation in the observation
        # given to the agent in order to get an association between these two
        # see also https://github.com/tensorflow/agents/issues/125#issuecomment-496583325
        observation_spec = {
            'state': TensorSpec((STATE_DIMENSIONS,), tf.float32),
            'mask': TensorSpec((ACTION_DIMENSIONS,), tf.float32)
        }

        layers = equal_spacing_fc(5)

        if actor_net is None:
            self.actor_net = MaskedActorNetwork(observation_spec, action_spec, layers)
        else:
            self.actor_net = actor_net

        if value_net is None:
            self.value_net = DummyMaskedValueNetwork(
                observation_spec, fc_layer_params=layers)
        else:
            self.value_net = value_net

        self.agent = tf_agents.agents.ppo.ppo_agent.PPOAgent(
            time_step_spec=ts.time_step_spec(observation_spec),
            action_spec=action_spec,

            actor_net=self.actor_net,
            value_net=self.value_net,

            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5),

            discount_factor=1,

            use_gae=True,
            use_td_lambda_return=True,
            lambda_value=0.85,

            num_epochs=30,

            # the observations are dicts { 'state': ..., 'mask': ... }
            # normalization does not make any sense for the mask
            normalize_observations=False,
        )


        if actor_net is not None or value_net is not None:
            self.agent.initialize()
        else:
            self.train_checkpointer = tf_agents.utils.common.Checkpointer(
                ckpt_dir=os.path.join(MODELS_PATH, self.name, 'Agent'), agent=self.agent)
            self.train_checkpointer.initialize_or_restore()

        # it seems like there is also agent.policy. I still don't understand when
        # one should use which and why but this one works for now.
        self.policy = self.agent.collect_policy

        # because tf_agents wants the data as trajectories
        # (prev_time_step, action, new_time_step), we have to store the prev_time_step
        # until we have the new_time_step to build the trajectory at which point
        # the new prev_time_step is the new_time_step
        # this variable is to keep track of the prev_time_step
        self.last_time_step = None

        # even though PPO is on policy, storing the stuff for a bit seems to be ok
        # and the examples in the tf_agents repo also use one
        self.replay_buffer = TFUniformReplayBuffer(self.agent.collect_data_spec,
            batch_size=1, max_length=REPLAY_BUFFER_SIZE)
        self.replay_buffer_position = 0

        self.clone_counter = 0

    def _add_trajectory(self, prev_time_step, action, new_time_step):
        """Add a trajectory (prev_time_step, action, new_time_step) to the replay buffer

        Also train the agent on the whole buffer if it is full.
        """

        traj = tf_agents.trajectories.trajectory.from_transition(
            prev_time_step, action, new_time_step)

        self.replay_buffer.add_batch(traj)
        self.replay_buffer_position += 1

        if self.replay_buffer_position == REPLAY_BUFFER_SIZE + 1:
            if not self.keep_models_fixed:
                self.agent.train(self.replay_buffer.gather_all())
            self.replay_buffer_position = 0
            self.replay_buffer.clear()

    def act(self, observation, valid_action_mask):
        observation = {
            'state': np.array(observation, dtype=np.float32),
            'mask': valid_action_mask
        }

        if self.last_time_step is None:
            # a new episode started
            self.last_time_step = _to_tf_timestep(ts.restart(observation))
            self.last_action_step = self.policy.action(self.last_time_step)
            return self.last_action_step.action.numpy()[0,0]

        new_time_step = _to_tf_timestep(ts.transition(observation, self.prev_reward))
        self._add_trajectory(self.last_time_step, self.last_action_step, new_time_step)

        self.last_time_step = new_time_step
        self.last_action_step = self.policy.action(new_time_step)
        self.prev_reward = None

        return self.last_action_step.action.numpy()[0,0]

    def observe(self, reward, terminal):
        if not terminal:
            self.prev_reward = reward
            return

        # even when the episode ends, tf_agents expects some observation
        # additionally to the reward. Because that makes no sense for us,
        # we just give it an observation consisting of all-zeros
        new_time_step = _to_tf_timestep(ts.termination({
            'state': np.zeros(STATE_DIMENSIONS),
            'mask': np.zeros(ACTION_DIMENSIONS)
        }, reward))

        self._add_trajectory(self.last_time_step, self.last_action_step, new_time_step)

        self.last_time_step = None
        self.last_action_step = None
        self.prev_reward = None

    def clone(self):
        """Return a clone of this agent with networks & predictor shared"""

        self.clone_counter += 1
        return TFAgentsPPOAgent(name=self.name + 'Clone' + str(self.clone_counter),
            actor_net=self.actor_net, value_net=self.value_net, predictor=self.predictor,
            keep_models_fixed=self.keep_models_fixed)

    def save_models(self, global_step: int):
        """Save actor, critic and predictor

        Args:
            global_step: the current game number, is appended to
                the filenames of the saved models
        """

        if self.keep_models_fixed:
            return

        super().save_models()
        self.train_checkpointer.save(global_step)
