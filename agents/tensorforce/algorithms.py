"""Contains implementations of TensorforceAgent for different algorithms"""

from tensorforce import agents

from agents.tensorforce.tensorforce_agent import TensorforceAgent
from agents.tensorforce import networks

# All of the agents below are currently trained with discount_factor = 0 and
# the metric used to judge them is how fast they learn
# not to play invalid cards

# The discount_factor = 0 is because we give a negative reward when
# they play some invalid card and that negative reward should be attributed
# to that card alone and not the cards from before

# It might be a good idea to start with discount_factor = 0 in general
# and increase it once the agents actually learned to play only valid cards
# (Though the default one of 0.99 is too high for a game which takes max 15 tricks)

# 1 Layer Proximal Policy Optimization
class TensorforcePPOAgent1L(TensorforceAgent):
    def build_agent(self, states, actions):
        return agents.PPOAgent(
            states,
            actions,
            networks.default_single_layer,
            discount=0
        )

# 2 Layer Proximal Policy Optimization
class TensorforcePPOAgent2L(TensorforceAgent):
    def build_agent(self, states, actions):
        return agents.PPOAgent(
            states=states,
            actions=actions,
            network=networks.default_2_layer,
            discount=0,
            step_optimizer=dict(type='adam', learning_rate=1e-5),
            baseline_mode='network',
            baseline=dict(type='custom', network=networks.default_2_layer),
            baseline_optimizer=dict(type='adam', learning_rate=1e-5),
        )

# 3 Layer Proximal Policy Optimization
class TensorforcePPOAgent3L(TensorforceAgent):
    def build_agent(self, states, actions):
        net = networks.equal_spacing(3)
        return agents.PPOAgent(
            states=states,
            actions=actions,
            network=net,
            discount=0,
            step_optimizer=dict(type='adam', learning_rate=1e-5),
            baseline_mode='network',
            baseline=dict(type='custom', network=net),
            baseline_optimizer=dict(type='adam', learning_rate=1e-5),
        )

# 2 Layer Vanilla Policy Gradient
class TensorforceVPGAgent2L(TensorforceAgent):
    def build_agent(self, states, actions):
        return agents.VPGAgent(
            states,
            actions,
            networks.default_2_layer,
            discount=0
        )

# 2 Layer Deep-Q-Networks
class TensorforceDQNAgent2L(TensorforceAgent):
    def build_agent(self, states, actions):
        return agents.DQNAgent(
            states,
            actions,
            networks.default_2_layer,
            discount=0
        )
