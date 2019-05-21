from tensorforce import agents

from agents.tensorforce.tensorforce_agent import TensorforceAgent
from agents.tensorforce import networks

# All of the agents below are currently trained with discount_factor = 0
# and the metric used to judge them is how fast they learn not to play
# cards they don't have on their hand

# The discount_factor = 0 is because we give a negative reward when
# they play some invalid card and that negative reward should be attributed
# to that card alone and not the cards from before

# It might be a good idea to start with discount_factor = 0 in general
# and increase it once the agents actually learned to play only valid cards
# (Though the default one of 0.99 is too high for a game which takes max 15 tricks)

# Proximal Policy Optimization
class TensorforcePPOAgent(TensorforceAgent):
    # Tried with small_single_layer network.
    # Percentage of valid cards which are played grows to 99% after 25k games
    # Slowly improves after that, maybe approx 99.7% after 40k games
    # (was somewhat hard to measure then)

    # Also tried with default_2_layer network. Training is slower
    # Percentage of valid cards which are played also improves but slower
    # => I'm not at 99% yet (only 50%).

    # Also just tried no_layer.
    # Percentage of valid cards goes up fast.
    # 98% after 4k games, takes a bit after that

    def __init__(self):
        super().__init__(agents.PPOAgent, networks.no_layer,
            discount=0, name='TensorforcePPOAgent0Layer')

# Vanilla Policy Gradient
class TensorforceVPGAgent(TensorforceAgent):
    # works, just not run long enough yet to get enough data

    def __init__(self):
        super().__init__(agents.VPGAgent,
            networks.default_2_layer, discount=0)

# Deep-Q-Networks
class TensorforceDQNAgent(TensorforceAgent):
    # Tried with small_single_layer network
    # Percentage of valid cards also improves but quite a lot slower than the PPOAgent
    # Didn't get it to 99%, took too long.

    def __init__(self):
        super().__init__(agents.DQNAgent,
            networks.default_2_layer, discount=0)

# Deep-Q-learning from demonstration
class TensorforceDQFDAgent(TensorforceAgent):
    # works, just not run long enough yet to get enough data

    def __init__(self):
        super().__init__(agents.DQFDAgent,
            networks.default_2_layer, discount=0)

# Normalized Advantage Functions
class TensorforceNAFAgent(TensorforceAgent):
    # Raises TensorForceError("Only unconstrained float actions valid for NAFModel.")
    # Haven't read about this yet but doesn't seem to be what we want given that error

    def __init__(self):
        super().__init__(agents.NAFAgent,
            networks.default_2_layer, discount=0)

# Trust Region Policy Optimization
class TensorforceTRPOAgent(TensorforceAgent):
    # Throws some exception. haven't looked that much into detail but I would
    # like to try this algorithm (or is PPO a strict upgrade ?)

    def __init__(self):
        super().__init__(agents.TRPOAgent,
            networks.default_2_layer, discount=0)
