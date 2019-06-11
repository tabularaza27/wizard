from game_engine.card import Card
from game_engine.player import Player

import numpy as np
import tensorflow.keras as K
from typing import List, Dict


class Featurizer:
    def transform(self, player: Player, trump: Card, first: Card, played: Dict[int, Card], players: List[Player],
                  played_in_round: Dict[int, List[Card]], color_left_indicator: np.ndarray):
        """
        Transforms the state into a numpy feature vector.

        Args:
            player: (Player) the player who is calling the featurizer.
            trump: (Card) trump card.
            first: (Card) first card.
            played: (dict(int, Card)) dict of played cards in the trick. The key is the index of the player
                who played that card. The card is None if no card was played from the player yet.
            players: (list(Players)) list of players in the game, includes THIS player.
            played_in_round: (dict(int, Card)) dict of played cards in the round for each player.
                The key is the index of the player who played that cards.
            color_left_indicator: (np.array) matrix with shape (players - 1, 4) which contains information
                about if a player has a certain suit left based on following the suit in past rounds.
                If the player i don't has color j left, then color_left_indicator[i, j] = 1, else it is 0.
        Returns:
            state: The state encoded into a numpy 1-d array.
        """
        raise  # Has to be implemented by child

    def state_dimension(self):
        """
        Returns:
            The length of the feature vector produced by this featurizer
        """
        raise  # Has to be implemented by child

    @staticmethod
    def cards_to_arr(cards):
        """
        Transforms cards into an array. All cards are binary (either in the
        list or not) except Z and N which are between 0 or 4. Indices are
        given by the Card.__int__ method. int(Z) == 53, int(N) == 52
        Args:
            cards: (list(Card)) list of cards to transform into array.

        Returns:
            arr: array (len==54) indicating the count of each card.
        """
        arr = np.zeros(Card.DIFFERENT_CARDS)
        for c in cards:
            if c is not None:
                arr[int(c)] += 1
        return arr

    @staticmethod
    def color_to_bin_arr(card):
        """
        Transforms a color into a one-hot encoding of it. The index order is
        given by card.Card.colors .
        Args:
            card: (Card) the card to extract color from.
            May be none.

        Returns:
            arr: one-hot encoding array of the color

        """
        bin_arr = [0] * len(Card.colors)
        if card is None:
            return bin_arr
        else:
            color = card.color
        index = Card.colors.index(color)
        bin_arr[index] = 1
        return bin_arr


class OriginalFeaturizer(Featurizer):
    """ Original Featurizer from https://github.com/mvelax/wizard-python"""

    def __init__(self, count_cards=True):
        self.count_cards = count_cards

    def transform(self, player: Player, trump: Card, first: Card, played: Dict[int, Card], players: List[Player],
                  played_in_round: Dict[int, List[Card]], color_left_indicator: np.ndarray):
        hand_arr = self.cards_to_arr(player.hand)

        trick_cards = filter(lambda card: not card is None, played.values())
        trick_arr = self.cards_to_arr(trick_cards)

        if self.count_cards:
            round_cards = sum(played_in_round.values(), [])
            game_arr = self.cards_to_arr(round_cards)
        else:
            game_arr = []
        trump_color = self.color_to_bin_arr(trump)
        first_color = self.color_to_bin_arr(first)
        player_score_win_predict = self.players_to_arr(players, player)

        return np.concatenate((hand_arr, trick_arr, game_arr, trump_color, first_color, player_score_win_predict))

    def state_dimension(self):
        return 180

    @staticmethod
    def players_to_arr(players, player):
        """
        Returns an array of the form [win1, predict1, ...,
        wini, predicti, winPersonal, predictPersonal]
        With the wins and predictions of each player finally with the
        wins and predictions of THIS player (player).
        -1 for any "non existent player".

        Args:
            players: list of all players
            player: THIS player

        Returns:
            arr: a list with the scores, wins, predictions of all players.
        """
        arr = []
        for other_player in players:
            if other_player == player:
                continue
            state = list(other_player.get_state())
            arr += state[1:]
        state = list(player.get_state())
        arr += state[1:]
        return arr


class FullFeaturizer(Featurizer):

    def transform(self, player: Player, trump: Card, first: Card, played: Dict[int, Card], players: List[Player],
                  played_in_round: Dict[int, List[Card]], color_left_indicator: np.ndarray):
        # cards in the hand of the player (players x 54)
        hand_arr = self.cards_to_arr(player.hand)

        # which cards where played in the current trick
        trick_arr = []
        for playerIndex, card in played.items():
            trick_arr.append(self.cards_to_arr([card]))
        trick_arr = np.concatenate(tuple(trick_arr))

        # one hot encoding of trump color (4)
        trump_color = self.color_to_bin_arr(trump)

        # how many cards of a certain color the player has left (4)
        player_color_left = np.zeros(4)
        for card in player.hand:
            if card.color != "White":
                player_color_left[Card.colors.index(card.color) - 1] += 1

        # which cards where played in the round by whom (players x 54)
        played_cards_arr = []
        for playerIndex, cards in played_in_round.items():
            played_cards_arr.append(self.cards_to_arr(cards))

        played_cards_arr = np.concatenate(tuple(played_cards_arr))

        # how much tricks the player predicted
        prediction = player.prediction

        player_index = -1

        # how many tricks the other players predicted (players - 1)
        other_predictions = []
        for index, p in enumerate(players):
            if p != player:
                other_predictions.append(p.prediction)
            else:
                player_index = index

        # the position of the player
        player_position = 0
        max_cards = max(len(cards) for cards in played_in_round.values())
        if not len(played_in_round[player_index]) == max_cards:
            for cards in played_in_round.values():
                if len(cards) == max_cards:
                    player_position += 1
        assert player_position < len(players)
        player_position_arr = K.utils.to_categorical(player_position, num_classes=len(players))
        tricks_left = len(player.hand)

        # indicator for how aggressive the player should try to get tricks
        playing_style = tricks_left - (prediction + sum(other_predictions))

        # How many tricks the player already achieved
        achieved_tricks = player.get_state()[1]

        # How many tricks the player still has to achieve
        tricks_needed = prediction - achieved_tricks

        feature_arr = np.concatenate(
            (hand_arr, trick_arr, trump_color, played_cards_arr, player_color_left, color_left_indicator.flatten(),
             np.array(other_predictions), player_position_arr,
             np.array([prediction, tricks_left, playing_style, achieved_tricks, tricks_needed])), axis=None)

        return feature_arr

    def state_dimension(self):
        return 519
