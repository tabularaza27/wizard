#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np
import sys
sys.path.append('../')

from game_engine.game import Game
from agents.rule_based_agent import RuleBasedAgent
from game_engine import player


games = 10000
players = [player.AverageRandomPlayer() for _ in range(3)]
players.append(RuleBasedAgent())
sum_scores = np.zeros(4)
wins = np.zeros(4)
print(sum_scores)
scores = []
for i in range(games):
    print("{}/{}".format(i, games))
    wiz = Game(players=players)
    scores = wiz.play_game()
    max = -1000
    max_index = -1
    for index, score in enumerate(scores):
        if score > max:
            max = score
            max_index = index
    wins[max_index] += 1
    sum_scores += scores
sum_scores /= games
print("Average Scores (Player 4 is Rule Based Agent):")
print(sum_scores)
print("Wins")
wins /= games
print(wins)


