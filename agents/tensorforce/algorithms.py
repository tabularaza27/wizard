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

# Proximal Policy Optimization
class TensorforcePPOAgent(TensorforceAgent):
    def __init__(self):
        super().__init__(agents.PPOAgent, networks.default_2_layer,
            discount=0)

# Vanilla Policy Gradient
class TensorforceVPGAgent(TensorforceAgent):
    def __init__(self):
        super().__init__(agents.VPGAgent,
            networks.default_2_layer, discount=0)

# Deep-Q-Networks
class TensorforceDQNAgent(TensorforceAgent):
    def __init__(self):
        super().__init__(agents.DQNAgent,
            networks.default_2_layer, discount=0)

# Deep-Q-learning from demonstration
class TensorforceDQFDAgent(TensorforceAgent):
    def __init__(self):
        super().__init__(agents.DQFDAgent,
            networks.default_2_layer, discount=0)

# The following agents do not work yet and throw some exception when used

# Normalized Advantage Functions
class TensorforceNAFAgent(TensorforceAgent):
    # TensorForceError("Only unconstrained float actions valid for NAFModel.")
    # Haven't read about this yet but might not be what we want given that error

    def __init__(self):
        super().__init__(agents.NAFAgent,
            networks.default_2_layer, discount=0)

# Trust Region Policy Optimization
class TensorforceTRPOAgent(TensorforceAgent):
    # Throws some exception. Haven't looked that much into detail
    # and it seems like PPO performs at least as good as this one in general
    # so maybe we don't even have to try it out

    def __init__(self):
        super().__init__(agents.TRPOAgent,
            networks.default_2_layer, discount=0)
