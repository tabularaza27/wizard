import glob
import os
from typing import List

import numpy as np

from agents.predictors import Predictor
from agents.original.featurizers import OriginalFeaturizer
from game_engine.player import Player, AverageRandomPlayer
from game_engine.card import Card

STATE_DIMENSIONS = 180
ACTION_DIMENSIONS = 4 * 13 + 2
MODELS_PATH = 'models/'

class RLAgent(AverageRandomPlayer):
    """Abstract RL Agent. Should be extended specific for each library.

    Attributes:
        name (str): Used for things like determining the folder
            where the model is saved. Defaults to class name.
        predictor (Predictor): A predictor specific to that agent.
            Doesn't share parameters with any other predictor.
        featurizer (OriginalFeaturizer): Used for getting the state
            from the arguments to play_card
        not_yet_given_reward (float | None):
            We want data in the following form (for use in libraries):
            self.act(state) -> self.observe(reward, terminal) -> self.act ...
            Because we don't know the reward or whether the game has ended
            directly after we play a card, we give this reward + terminal info
            before each card (starting from the second one) or when the game
            has finished so that there is a one-to-one association
            from act to observe.
            Therefore we have to keep track of whether there is a reward
            which we should give before we are asked for the next card
            which is what this variable is for
    """

    def __init__(self, name=None, predictor=None):
        super().__init__()

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

        predictor_model_path = os.path.join(MODELS_PATH, self.name, 'Predictor/')
        if not os.path.exists(predictor_model_path):
            os.makedirs(predictor_model_path)

        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = Predictor(model_path=predictor_model_path)

        self.featurizer = OriginalFeaturizer()
        self.not_yet_given_reward = None

        # TODO remove this when the plotting functionality is there,
        # should be able to handle this easier
        # Tracks how much of the last 10000 cards which the agent wanted to play
        # were actually valid cards -> gives us an estimate on how good the agent
        # actually understands the rules
        self.last_10000_cards_played_valid = []
        self.valid_rate = 0

    def save_models(self):
        self.predictor.save_model()

    def act(self, state: np.array) -> Card:
        """Returns the action the model takes if it is in the given state

        Args:
            state: The observation (output from featurizer)

        Returns: The card the model will play (removed from hand)
        """
        raise # Has to be implemented by child

    def observe(self, reward: float, terminal: bool):
        """Feeds the reward to the model & resets stuff if terminal is True"""
        raise # Has to be implemented by child

    def play_card(self, trump: Card, first: Card, played: List[Card],
            players: List[Player], played_in_game: List[Card]):
        """Plays a card based on the agents action"""

        # TODO replace this print with the corresponding plotting once
        # implemented to keep track of how many valid cards the agent plays
        if len(self.last_10000_cards_played_valid) == 10000:
            self.valid_rate = np.mean(self.last_10000_cards_played_valid)
            self.last_10000_cards_played_valid = []

        if self.not_yet_given_reward is not None:
            self.observe(reward=self.not_yet_given_reward, terminal=False)
            self.not_yet_given_reward = None

        state = self.featurizer.transform(self, trump, first,
            played, players, played_in_game)
        action = self.act(state)

        # find the card which corresponds to that action and return it if valid
        playable_cards = self.get_playable_cards(first)
        for card in self.hand:
            if int(card) == action:
                if card not in playable_cards:
                    break
                self.hand.remove(card)
                self.not_yet_given_reward = 0
                self.last_10000_cards_played_valid.append(1)
                return card

        # the agent is trying to play a card which is invalid
        # we give him a negative reward, play a random card and continue
        self.last_10000_cards_played_valid.append(0)
        self.not_yet_given_reward = -100
        return super().play_card(trump, first, played, players, played_in_game)

    def get_prediction(self, trump: Card, num_players: int):
        """Return the round prediction using the build in NN Predictor"""
        self.prediction_x, prediction = \
            self.predictor.make_prediction(self.hand, trump)
        return prediction

    def announce_result(self, num_tricks_achieved: int, reward: float):
        """use the game result as a feedback for the agent and predictor"""
        super().announce_result(num_tricks_achieved, reward)
        reward_to_give = self.not_yet_given_reward + reward
        self.observe(reward=reward_to_give, terminal=True)
        self.not_yet_given_reward = None
        self.predictor.add_game_result(self.prediction_x, num_tricks_achieved)
