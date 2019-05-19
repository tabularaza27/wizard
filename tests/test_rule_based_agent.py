#!/usr/bin/env python3

"""Plays the rule based agent against three average random players over 10,000 games.
The average points and win percentages are then printed."""

import numpy as np
# This allows the file to be run in a test folder as opposed to having to be in the root directory
import sys
sys.path.append('../')

from game_engine.game import Game
from agents.rule_based_agent import RuleBasedAgent
from game_engine import player

# Can change the amount of games if you want
games = 10000
players = [player.AverageRandomPlayer() for _ in range(3)]
players.append(RuleBasedAgent())
sum_scores = np.zeros(4)
wins = np.zeros(4)

# plays the games
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games), flush=True)
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
    # Aggregates the scores from all previous games
    sum_scores += scores

sum_scores /= games
print("Average Scores (Player 4 is Rule Based Agent):")
print(sum_scores)
print("Win Percentage")
wins /= games
print(wins)


