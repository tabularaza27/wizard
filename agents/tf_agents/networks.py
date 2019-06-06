import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents.networks

from tensorflow.python.keras.engine.network import Network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork

class MaskedActorNetwork(ActorDistributionNetwork):
    def __init__(self, input_tensor_spec, output_tensor_spec, fc_layer_params):
        super().__init__(input_tensor_spec['state'], output_tensor_spec, fc_layer_params)

    def call(self, observations, step_type, network_state):
        states = observations['state']
        masks = observations['mask']

        action_distributions, new_network_states = super().call(
            states, step_type, network_state)

        # for some reason, when we get a batch there is an extra axis of dimension 1
        # we therefore also have to insert this axis into our mask
        if (len(action_distributions.logits.shape) == 4):
            masks = tf.expand_dims(masks, 2)

        masked_logits = masks + action_distributions.logits

        # the dtype doesn't refer to the logits
        # but the action that is then created from the distribution
        return tfp.distributions.Categorical(
            logits=masked_logits, dtype=tf.int64), new_network_states

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)

# the value network gets the same input as the actor network
# however only the actor network actually needs the mask
# so in the value network, we have to throw it away explicitly
class DummyMaskedValueNetwork(ValueNetwork):
    def __init__(self, input_tensor_spec, fc_layer_params):
        super().__init__(input_tensor_spec['state'], fc_layer_params)

    def call(self, observation, step_type=None, network_state=()):
        return super().call(observation['state'], step_type, network_state)

    def __call__(self, inputs, *args, **kwargs):
        return super(Network, self).__call__(inputs, *args, **kwargs)
