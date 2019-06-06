import numpy as np
import tensorflow as tf
import tf_agents
import tf_agents.agents
import tf_agents.trajectories.time_step as ts
import tf_agents.replay_buffers

from tensorflow.contrib.framework import TensorSpec, BoundedTensorSpec
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from agents.rl_agent import RLAgent, STATE_DIMENSIONS, ACTION_DIMENSIONS
from agents.tf_agents.layers import equal_spacing_fc
from agents.tf_agents.networks import MaskedActorNetwork, DummyMaskedValueNetwork

REPLAY_BUFFER_SIZE = 5000

class TFAgentsPPOAgent(RLAgent):
    def __init__(self, name=None, actor_net=None, value_net=None, predictor=None):
        super().__init__(name, predictor)

        action_spec = BoundedTensorSpec((1,), tf.int64, 0, ACTION_DIMENSIONS - 1)
        observation_spec = {
            'state': TensorSpec((STATE_DIMENSIONS,), tf.float32),
            'mask': TensorSpec((ACTION_DIMENSIONS,), tf.float32)
        }

        layers = equal_spacing_fc(2)

        if actor_net is None:
            actor_net = MaskedActorNetwork(observation_spec, action_spec, layers)
        if value_net is None:
            value_net = DummyMaskedValueNetwork(observation_spec, fc_layer_params=layers)
        self.actor_net = actor_net
        self.value_net = value_net

        self.agent = tf_agents.agents.ppo.ppo_agent.PPOAgent(
            time_step_spec=ts.time_step_spec(observation_spec),
            action_spec=action_spec,

            actor_net=actor_net,
            value_net=value_net,

            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4),

            discount_factor=0.8,

            use_gae=True,
            use_td_lambda_return=True,
            lambda_value=0.8,

            num_epochs=25,

            # the observations are dicts { 'state': ..., 'mask': ... }
            # normalization does not make any sense for the mask
            normalize_observations=False,
        )

        self.agent.initialize()
        self.policy = self.agent.collect_policy

        self.last_time_step = None

        self.replay_buffer = TFUniformReplayBuffer(self.agent.collect_data_spec,
            batch_size=1, max_length=REPLAY_BUFFER_SIZE)
        self.replay_buffer_position = 0

    def _to_tf_timestep(self, time_step):
        time_step = tf_agents.utils.nest_utils.batch_nested_array(time_step)
        return tf.contrib.framework.nest.map_structure(tf.convert_to_tensor, time_step)

    def _add_trajectory(self, prev_time_step, action, new_time_step):
        traj = tf_agents.trajectories.trajectory.from_transition(
            prev_time_step, action, new_time_step)

        self.replay_buffer.add_batch(traj)
        self.replay_buffer_position += 1

        if self.replay_buffer_position == REPLAY_BUFFER_SIZE + 1:
            self.agent.train(self.replay_buffer.gather_all())
            self.replay_buffer_position = 0
            self.replay_buffer.clear()

    def act(self, observation, valid_action_mask):
        observation = {
            'state': np.array(observation, dtype=np.float32),
            'mask': valid_action_mask
        }

        if self.last_time_step is None:
            self.last_time_step = self._to_tf_timestep(ts.restart(observation))
            self.last_action_step = self.policy.action(self.last_time_step)
            return self.last_action_step.action.numpy()[0,0]

        new_time_step = self._to_tf_timestep(ts.transition(observation, self.prev_reward))
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
        new_time_step = self._to_tf_timestep(ts.termination({
            'state': np.zeros(STATE_DIMENSIONS),
            'mask': np.zeros(ACTION_DIMENSIONS)
        }, reward))

        self._add_trajectory(self.last_time_step, self.last_action_step, new_time_step)

        self.last_time_step = None
        self.last_action_step = None
        self.prev_reward = None

    def clone(self):
        return TFAgentsPPOAgent(name=self.name + 'Clone' + str(np.random.randint(1e10)),
            actor_net=self.actor_net, value_net=self.value_net, predictor=self.predictor)
