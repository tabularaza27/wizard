#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np
# This allows the file to be run in a test folder as opposed to having to be in the root directory
import sys
sys.path.append('../')

from game_engine.game import Game
from agents.rl_agents import RLAgent
from agents.rule_based_agent import RuleBasedAgent
from game_engine import player


games = 2000
seed(2)
players = [RuleBasedAgent() for _ in range(3)]
players.append(RLAgent())
wins = np.zeros(4)
players[0].load_estimator()
# for rl_player in range(3):
#     players.append(RLAgent(estimator=players[0].estimator))
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games))
    wiz = Game(players=players)
    scores = wiz.play_game()
    # Determines the winner of the game
    max = -1000
    max_index = -1
    for index, score in enumerate(scores):
        if score > max:
            max = score
            max_index = index
    wins[max_index] += 1
players[3].save_estimator()
players[3].predictor.save_model()
# scores = np.array(scores)
print("Win Percentage")
wins /= games
print(wins)