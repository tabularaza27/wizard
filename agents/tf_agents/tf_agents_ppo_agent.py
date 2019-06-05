#!/usr/bin/env python3

import time
import numpy as np
import tensorflow as tf
import tf_agents
import tf_agents.networks
import tf_agents.agents
import tf_agents.trajectories.time_step as ts
import tf_agents.replay_buffers

from agents.rl_agent import RLAgent

STATE_DIMENSIONS = 180
ACTION_DIMENSIONS = 4 * 13 + 2
LAYERS = (138, 96)

class TFAgentsPPOAgent(RLAgent):
    def __init__(self, name=None, actor_net=None, value_net=None, predictor=None):
        super().__init__(name, predictor)

        # TODO check if observation_spec can be made unbounded
        observation_spec = tf.contrib.framework.BoundedTensorSpec(
            (STATE_DIMENSIONS,), tf.float32, -1000, 1000)
        action_spec = tf.contrib.framework.BoundedTensorSpec((1,), tf.int64, 0, ACTION_DIMENSIONS - 1)

        if actor_net is None:
            actor_net = tf_agents.networks.actor_distribution_network.ActorDistributionNetwork(
                observation_spec, action_spec, fc_layer_params=LAYERS)
        if value_net is None:
            value_net = tf_agents.networks.value_network.ValueNetwork(
                observation_spec, fc_layer_params=LAYERS)
        self.actor_net = actor_net
        self.value_net = value_net

        self.agent = tf_agents.agents.ppo.ppo_agent.PPOAgent(
            time_step_spec=ts.time_step_spec(observation_spec),
            action_spec=action_spec,

            actor_net=actor_net,
            value_net=value_net,

            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4),

            discount_factor=0,

            use_gae=True,
            use_td_lambda_return=True,
            lambda_value=0.8,

            num_epochs=25,
        )

        self.agent.initialize()
        self.policy = self.agent.collect_policy

        self.last_time_step = None

        self.replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec, batch_size=1, max_length=15 * 1000)
        self.replay_buffer_position = 0

    def _to_tf_timestep(self, time_step):
        time_step = tf_agents.utils.nest_utils.batch_nested_array(time_step)
        return ts.TimeStep(*[tf.convert_to_tensor(e) for e in tf.nest.flatten(time_step)])

    def _add_trajectory(self, prev_time_step, action, new_time_step):
        traj = tf_agents.trajectories.trajectory.from_transition(
            prev_time_step, action, new_time_step)
        self.replay_buffer.add_batch(traj)

        self.replay_buffer_position += 1
        if self.replay_buffer_position == 5000:
            trajectories = self.replay_buffer.gather_all()
            self.agent.train(trajectories)
            self.replay_buffer_position = 0
            self.replay_buffer.clear()

    def act(self, observation):
        observation = np.array(observation, dtype=np.float32)

        if self.last_time_step is None:
            self.last_time_step = self._to_tf_timestep(ts.restart(observation))
            self.last_action_step = self.policy.action(self.last_time_step)
            return self.last_action_step.action.numpy()[0,0]

        prev_time_step = self.last_time_step
        prev_action_step = self.last_action_step
        prev_reward = self.prev_reward

        new_time_step = self._to_tf_timestep(ts.transition(observation, prev_reward))

        self._add_trajectory(prev_time_step, prev_action_step, new_time_step)

        new_action_step = self.policy.action(new_time_step)

        self.last_time_step = new_time_step
        self.last_action_step = new_action_step
        self.prev_reward = None

        return new_action_step.action.numpy()[0,0]

    def observe(self, reward, terminal):
        if not terminal:
            self.prev_reward = reward
            return

        prev_time_step = self.last_time_step
        prev_action_step = self.last_action_step

        new_time_step = self._to_tf_timestep(ts.termination(np.zeros(STATE_DIMENSIONS), reward))

        self._add_trajectory(prev_time_step, prev_action_step, new_time_step)

        self.last_time_step = None
        self.last_action_step = None
        self.prev_reward = None

    def clone(self):
        return TFAgentsPPOAgent(name=self.name + 'Clone' + str(np.random.randint(1e10)),
            actor_net=self.actor_net, value_net=self.value_net, predictor=self.predictor)
