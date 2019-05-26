import os

from tensorforce.agents.agent import Agent

from agents import rl_agent

class TensorforceAgent(rl_agent.RLAgent):
    """Base class for all agents which use tensorforce-agents internally.

    Attributes:
        agent (tensorforce.agents.agent.Agent):
            The tensorforce algorithm to use for this specific agent
        agent_model_path (str): The directory where all the files
            for the tensorforce agent model will be saved
    """

    def __init__(self, name=None):
        super().__init__(name)

        self.agent = self.build_agent(
            states=dict(type='float', shape=(rl_agent.STATE_DIMENSIONS,)),
            actions=dict(type='int', num_actions=rl_agent.ACTION_DIMENSIONS)
        )

        self.agent_model_path = os.path.join(rl_agent.MODELS_PATH,
            self.name, 'Agent/')
        if os.path.exists(self.agent_model_path):
            self.agent.restore_model(self.agent_model_path)
        else:
            os.makedirs(self.agent_model_path)

    def build_agent(self, states: dict, actions: dict) -> Agent:
        """Build a new tensorforce-agent used for this agent.

        Args:
            states: specifies the shape in tensorforce form
                used in the tensorforce-agent
            actions: similary specifies the actionspace

        Returns: The tensorforce-agent used for playing tricks
        """

        raise # Should be overwritten by child class

    def save_models(self):
        super().save_models()
        self.agent.save_model(self.agent_model_path)

    def observe(self, reward, terminal):
        self.agent.observe(reward=reward, terminal=terminal)

    def act(self, state):
        return self.agent.act(state)
