#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np

from game_engine.game import Game
from agents.rl_agents import RLAgent
from game_engine import player


games = 2000
seed(2)
players = [RLAgent() for _ in range(2)]
# players[0].load_estimator()
# for rl_player in range(3):
#     players.append(RLAgent(estimator=players[0].estimator))
players.append(player.AverageRandomPlayer())
players.append(player.RandomPlayer())
scores = []
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games))
    wiz = Game(players=players)
    scores.append(wiz.play_game())
players[0].save_estimator()
players[0].predictor.save_model()
scores = np.array(scores)
print("Done")
