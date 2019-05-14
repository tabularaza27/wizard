"""Implements Player Base class and its subclasses RandomPlayer and AverageRandomPlayer"""

import random
import collections

from game_engine.card import Card


class Player:
    """Player Base Class

    Attributes:
        hand (:obj: `list` of :obj: `Card`): list of cards in players hand
        score (int): current score of the player
        reward (int):
        wins (int): # of won tricks by player
        prediction (int): prediction for current round


    """

    def __init__(self):
        self.hand = []
        self.score = 0
        self.reward = 0
        self.wins = 0
        self.prediction = -1

    def get_playable_cards(self, first):
        """Determines the currently possible cards that player can play, based on his hand and the suit of the first
        card of the trick

        Args:
            first (Card): first card, that has been played in the current trick.

        Returns:
            list of Cards that player can play
        """
        playable_cards = []
        first_colors = []
        if first is None:
            return self.hand
        for card in self.hand:
            # White cards can ALWAYS be played.
            if card.color == "White":
                playable_cards.append(card)
            # First card color can ALWAYS be played.
            elif card.color == first.color:
                first_colors.append(card)
            # Other colors can only be played if there
            # no cards of the first color in the hand.
        if len(first_colors) > 0:
            return playable_cards + first_colors
        else:
            # Cannot follow suit, use ANY card.
            return self.hand

    def play_card(self, trump, first, played, players, played_in_game):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def get_prediction(self, trump, num_players):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def get_trump_color(self):
        raise NotImplementedError("This needs to be implemented by your Player class")

    def announce_result(self, num_tricks_achieved, reward):
        self.wins = num_tricks_achieved
        self.reward = reward
        self.score += reward
        self.hand = []

    def get_state(self):
        return self.score, self.wins, self.prediction

    def reset_score(self):
        self.score = 0


class RandomPlayer(Player):
    """A completely random agent, it always chooses all its actions randomly"""

    def __init__(self):
        super().__init__()

    def play_card(self, trump, first, played_in_round, players, played_in_game):
        """Randomly play any VALID card.

        Args:
            trump (Card): trump of the current round
            first (Card): first card played in the current trick
            played_in_round (:obj: `list` of `Card`): list of cards played in the round. Not used in random player
            players (:obj: `list` of `Player`): list of players in the game. Not used in random player
            played_in_game (:obj: `list` of `Card`): list of cards played in the game. Not used in random player

        Returns:
            Card: the chosen card from the player hand to play in the current trick
            """
        possible_actions = super().get_playable_cards(first)
        if not isinstance(possible_actions, list):
            possible_actions = list(possible_actions)
        random.shuffle(possible_actions)
        card_to_play = possible_actions[0]
        self.hand.remove(card_to_play)
        # print("Playing card {} from {}".format(card_to_play, self.hand))
        return card_to_play

    def get_prediction(self, trump, num_players):
        """Randomly return any number of wins between 0 and total number of games.

        Returns:
            int: number of wins predicted for the round
        """
        prediction = random.randrange(len(self.hand))
        self.prediction = prediction
        return prediction

    def get_trump_color(self):
        """Randomly chooses trump color from cards of own hand

        Returns:
            str: color of trump
        """
        return random.choice(Card.colors[1:])


class AverageRandomPlayer(RandomPlayer):
    """Agent that uses random cards, but chooses an 'average'
    prediction of wins and a trump color corresponding to
    the color the agent has the most of in its hand."""

    def __init__(self):
        super().__init__()

    def get_prediction(self, trump, num_players):
        """Predicts number of tricks corresponding to: #card on hand / # players

        Args:
            trump (Card): not used for this agent
            num_players (int): number of players in the game

        Returns:
            int: number of predicted trick by player for current round
        """
        prediction = len(self.hand) // num_players
        self.prediction = prediction

        return prediction

    def get_trump_color(self):
        """Determines trump color by choosing the color the agent has the most of in its hand

        Returns:
            str: color of trump
        """
        color_counter = collections.Counter()
        for card in self.hand:
            color = card.color
            if color == "White":
                continue
            color_counter[color] += 1
        if not color_counter.most_common(1):
            return super().get_trump_color()
        else:
            return color_counter.most_common(1)[0][0]
