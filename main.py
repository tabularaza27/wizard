#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np

from game_engine.game import Game
from agents.original.rl_agents import OriginalRLAgent
from game_engine import player


games = 2000
seed(2)
players = [OriginalRLAgent()]
# players[0].load_estimator()
for rl_player in range(1):
    players.append(OriginalRLAgent(
        estimator=players[0].estimator,
        predictor=players[0].predictor))
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
