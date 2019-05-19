#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np
import sys
sys.path.append('../')

from game_engine.card import Card
from game_engine.deck import Deck
from agents.rule_based_agent import RuleBasedAgent
# from game_engine import player

joker = Card('White', 14)
deck = Deck()
card = Deck.draw(deck)[0]
played = Deck.draw(deck, 3)
trump = Deck.draw(deck)[0]
print(played)
player = RuleBasedAgent()
print(card)
print(trump)
# print(joker)
print(player.win_probability(played, card, trump, played[0], range(4)))
print(player.number_of_stronger_cards_remaining(card, trump, played[0], played))
# games = 1
# seed(10)
# players = [player.AverageRandomPlayer() for _ in range(3)]
# players.append(RuleBasedAgent())
# scores = []
# for i in range(games):
#     print("{}/{}".format(i, games))
#     wiz = Game(players=players)
#     scores.append(wiz.play_game())
# scores = np.array(scores)
# print(scores)
