# import random
import collections

from game_engine import player


class RuleBasedAgent(player.Player):
    """A computer player that makes decision on predefined rules.
    Aims to resemble the performance and behaviour of that of a human."""

    def __init__(self):
        super().__init__()
        self.played = []

    def get_prediction(self, trump, num_players):
        """
        This algorithm predicts the amount of tricks that the player should win. It does this by using a weighted average between the
        expected amount of tricks calculated as the amount of cards divide by the amount of players and another algorithm that assigns
        an expected return for each card and sums all the expected returns together.
        :param trump:
        :param num_players:
        :return: prediction:
        """
        self.played = []
        weight = 0.7
        prediction = 0
        for card in self.hand:
            if card.value == 14:
                prediction += 0.95
            elif card.color == trump:
                if card.value > 10:
                    prediction += 0.85
                elif card.value > 5:
                    prediction += 0.5
                else:
                    prediction += 0.35
            else:
                if card.value > 10:
                    prediction += 0.35
                elif card.value > 5:
                    prediction += 0.15
                else:
                    prediction += 0.1
        prediction = weight * prediction + (1-weight) * len(self.hand) // num_players
        prediction = round(self.bound(prediction,len(self.hand), 0), 0)
        self.prediction = prediction
        return prediction

    def announce_result(self, num_tricks_achieved, reward):
        self.wins = num_tricks_achieved
        self.reward = reward
        self.score += reward
        self.hand = []

    def play_card(self, trump, first, played, players, played_in_game):
        """
        Finds the card whose win probability most closely matches that of the win desirability
        :param trump:
        :param first:
        :param played:
        :param players:
        :param played_in_game:
        :return: best_card
        """
        for card in played:
            self.played.append(card)
        win_desirability = self.win_desirability(players)
        best_card = self.get_playable_cards(first)[0]
        best_delta = abs(win_desirability - self.win_probability(played, best_card, trump, first, players))
        best_win_likelihood = 0
        for card in self.get_playable_cards(first):
            win_likelihood = self.win_probability(played, card, trump, first, players)
            delta = abs(win_desirability - win_likelihood)
            if delta < best_delta:
                best_card = card
                best_delta = delta
            if win_likelihood > best_win_likelihood: best_win_likelihood = win_likelihood
        # If going to lose anyway then play the worst card (the one with the greatest number of stronger cards)
        if win_likelihood == 0:
            counter = 0
            for card in self.get_playable_cards(first):
                if self.number_of_stronger_cards_remaining(card, trump, first, played) > 0: best_card = card
        self.hand.remove(best_card)
        return best_card

    def get_trump_color(self):
        """
        Determines trump color by choosing the color the agent has the most of in its hand
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

    def win_probability(self, played, card, trump, first, players):
        """
        Given a card and the current state, this algorithm calculates the probability of winning by seeing if the card is
        stronger than those already played and estimates the chance that a stronger card will be played in the future.
        :param played:
        :param card:
        :param trump:
        :param first:
        :param players:
        :return: probability of winning
        """
        if first is None:
            return ((60 - len(played)) - self.number_of_stronger_cards_remaining(card, trump, first, played))/ (60 - len(played))
        else:
            for other_card in played:
                if self.stongest_card(other_card, card, trump, first) == other_card:
                    return 0
            if len(played) == len(players) - 1:
                return 1
            else:
                return ((60 - len(played)) - self.number_of_stronger_cards_remaining(card, trump, first, played))/ (60 - len(played))

    def win_desirability(self, players):
        """
        Approximates the desire to win based on how many rounds are left to play and how many predictions are left to make
        and also dependent on how many tricks left for the opposition to make
        :param players:
        :return: win desirability
        """
        if (self.prediction - self.wins) >= len(self.hand):
            return 1
        elif self.prediction <= self.wins:
            return 0
        else:
            desirability = (self.prediction - self.wins)/len(self.hand)
            for player in players:
                if player != self:
                    desirability += (1/(len(players)+1))*(player.prediction - player.wins)/len(self.hand)
            return self.bound(desirability, 1, 0)

    def bound(self, value, max, min):
        """
        Bounds the value between a given range.
        :param value:
        :param max:
        :param min:
        :return: a value between the maximum and minimum
        """
        if value > max:
            return max
        elif value < min:
            return min
        else:
            return value

    def cards_left_by_color(self, color):
        """
        Keeps track of all cards played in that round and deduces how many cards of that color are left
        :param color:
        :return: number_of_cards_of_that_color_left
        """
        trump_cards_left = []
        for card in self.played:
            if card.color == color:
                trump_cards_left.append(card)
        return trump_cards_left

    def number_of_stronger_cards_remaining(self, card, trump, first, played):
        """
        Estimates the number of stronger cards remaining in the deck
        :param card:
        :param trump:
        :param first:
        :param played:
        :return: number of stronger cards
        """
        counter = 0
        if card.value == 14: return counter
        played_trump_counter = 0
        for played_card in self.cards_left_by_color(trump):
            if played_card.value > card.value: played_trump_counter += 1
        played_second_trump_counter = 0
        if first is None:
            color = card.color
        else:
            color = first.color
        for played_card in self.cards_left_by_color(color):
            if played_card.value > card.value: played_second_trump_counter += 1
        if card.color == trump:
            return (13 - played_trump_counter - card.value) + 4
        elif first is not None and card.color == first.color:
            return (13 - played_second_trump_counter - card.value) + 17 - played_trump_counter
        else:
            return ((13 - card.value) * 2) + 30 - played_trump_counter - played_second_trump_counter

    def stongest_card(self, card1, card2, trump, first):
        """
        Determines which card has precedence given the trump and the first card.
        :param card1:
        :param card2:
        :param trump:
        :param first:
        :return: card1 or card2
        """
        if card1.value == 14:
            return card1
        if card2.value == 14:
            return card2
        if card1.color == trump:
            if card2.color == trump:
                if card1.value > card2.value:
                    return card1
                else:
                    return card2
            else:
                return card1
        elif first is not None and card1.color == first.color:
            if card2.color == trump:
                return card2
            elif card2.color == first.color:
                if card1.value > card2.value:
                    return card1
                else:
                    return card2
            else:
                return card1
        else:
            if card2.color == trump:
                return card2
            elif first is not None and card2.color == first.color:
                return card2
            else:
                if card1.value > card2.value:
                    return card1
                else:
                    return card2




