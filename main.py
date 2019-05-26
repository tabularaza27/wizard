#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np

from game_engine.game import Game
from agents.tensorforce.algorithms import TensorforcePPOAgent2L
from game_engine.player import AverageRandomPlayer

games = 2000
seed(2)
players = [TensorforcePPOAgent2L(), AverageRandomPlayer(),
    AverageRandomPlayer(), AverageRandomPlayer()]
scores = []
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games))
    wiz = Game(players=players)
    scores.append(wiz.play_game())
players[0].save_models()
scores = np.array(scores)
print("Done")
