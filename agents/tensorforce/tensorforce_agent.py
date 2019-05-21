import os

from agents import rl_agent

class TensorforceAgent(rl_agent.RLAgent):
    def __init__(self, TFAgent, network, name=None, **kwargs):
        super().__init__(name)

        self.agent = TFAgent(
            states=dict(type='float', shape=(rl_agent.STATE_DIMENSIONS,)),
            actions=dict(type='int', num_actions=rl_agent.ACTION_DIMENSIONS),
            network=network,
            **kwargs
        )

        self.agent_model_path = os.path.join(rl_agent.MODELS_PATH,
            self.name, 'Agent/')
        if os.path.exists(self.agent_model_path):
            self.agent.restore_model(self.agent_model_path)
        else:
            os.makedirs(self.agent_model_path)

    def save_models(self):
        super().save_models()
        self.agent.save_model(self.agent_model_path)

    def observe(self, reward, terminal):
        self.agent.observe(reward=reward, terminal=terminal)

    def act(self, state):
        return self.agent.act(state)
