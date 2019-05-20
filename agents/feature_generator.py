from game_engine.card import Card
from game_engine.player import Player

import numpy as np
from typing import List, Dict


class FeatureGenerator(object):

    def compute_features(self, player: Player, players: List[Player], trump_color: str, trick_cards: List[Card],
                         played_cards: Dict[List[Card], int], color_left_indicator: np.ndarray):
        """
        Computes a numpy feature vector for a game state specified through the different parameters.

        Arguments:
            player: The player for whom the features should be computed.
            players: All players in the game.
            trump_color: Name of the trump color, None if there is no trump color.
            trick_cards: The cards which were played in the current trick.
            played_cards: The cards which were played in the current round.
            color_left_indicator: boolean array which encodes for every player,
                if he has the color left or not ((players - 1) x 4)

        Returns:
            1-dimensional numpy array which holds all features.
        """

        # cards in the hand of the player (players x 54)
        hand_arr = self.cards_to_arr(player.hand)

        # which cards where played in the current trick
        trick_arr = self.cards_to_arr(trick_cards)

        # one hot encoding of trump color (4)
        trump_color = self.color_to_bin_arr(trump_color)

        # how many cards of a certain color the player has left (4)
        player_color_left = np.zeros(4)
        for card in player.hand:
            if card.color != "White":
                player_color_left[Card.colors.index(card.color) - 1] += 1

        # which cards where played in the round by whom (players x 54)
        played_cards_arr = []
        for playerIndex, cards in enumerate(played_cards):
            played_cards_arr.append(self.cards_to_arr(cards))
        played_cards_arr = np.concatenate(tuple(played_cards_arr))

        # how much tricks the player predicted
        prediction = player.prediction

        # how many tricks the other players predicted (players - 1)
        other_predictions = []
        for p in players:
            if p != player:
                other_predictions.append(p.prediction)

        tricks_left = len(player.hand)

        # indicator for how aggressive the player should try to get tricks
        playing_style = tricks_left - (prediction + sum(other_predictions))

        return np.concatenate(
            (hand_arr, trick_arr, trump_color, played_cards_arr, player_color_left, color_left_indicator,
             np.array(other_predictions),
             np.array([prediction, tricks_left, playing_style])))

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
            arr[int(c)] += 1
        return arr

    @staticmethod
    def color_to_bin_arr(color):
        """
        Transforms a color into a one-hot encoding of it. The index order is
        given by card.Card.colors .
        Args:
            card: (Card) the card to extract color from.
            May be none.

        Returns:
            arr: one-hot encoding array of the color

        """
        bin_arr = np.zeros(Card.colors)
        if color is None:
            return bin_arr
        index = Card.colors.index(color)
        bin_arr[index] = 1
        return bin_arr
