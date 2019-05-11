#!/usr/bin/env python3

from random import seed

import numpy as np

from Agents.RLAgents import RLAgent
from Game_Engine.Wizard import Wizard
from Game_Engine import Player

games = 2000
seed(2)
players = [RLAgent() for _ in range(4)]
# players[0].load_estimator()
# for rl_player in range(3):
#     players.append(RLAgent(estimator=players[0].estimator))
players.append(Player.AverageRandomPlayer())
players.append(Player.RandomPlayer())
scores = []
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games))
    wiz = Wizard(players=players)
    scores.append(wiz.play())
players[0].save_estimator()
scores = np.array(scores)
print("Done")
