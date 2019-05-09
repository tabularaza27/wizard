import random

from game_engine.card import Card


class Deck(object):

    def __init__(self):
        self.cards = []
        # Add four colors with 1-13 cards.
        for val in range(1, 14):
            for color in Card.colors[1:]:
                self.cards.append(Card(color, val))
        # Add four Zs (white, 14) and four Ns (white, 0)
        for _ in range(4):
            self.cards.append(Card("White", 0))
            self.cards.append(Card("White", 14))
        random.shuffle(self.cards)

    def draw(self, num=1):
        """
        Returns a list of the drawn cards from the deck.
        Removes the card from the deck.
        :param num: int Number of cards to draw.
        :return: list: Cards drawn
        """
        drawn = []
        for _ in range(num):
            drawn.append(self.cards.pop())
        return drawn

    def is_empty(self):
        return len(self.cards) <= 0