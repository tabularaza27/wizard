#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np

from game_engine.game import Game
from agents.rule_based_agent import RuleBasedAgent
from game_engine import player


games = 1
seed(10)
players = [player.AverageRandomPlayer() for _ in range(3)]
players.append(RuleBasedAgent())
scores = []
for i in range(games):
    print("{}/{}".format(i, games))
    wiz = Game(players=players)
    scores.append(wiz.play_game())
scores = np.array(scores)
print(scores)
