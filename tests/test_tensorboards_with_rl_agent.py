#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np
# This allows the file to be run in a test folder as opposed to having to be in the root directory
import sys
sys.path.append('../')

import tensorflow as tf
import os
import threading
from game_engine.game import Game
from agents.rl_agent import RLAgent
from agents.tensorforce.algorithms import TensorforcePPOAgent1L, TensorforcePPOAgent3L
from game_engine.player import AverageRandomPlayer
from agents.rule_based_agent import RuleBasedAgent


def launch_tensor_board():
    os.system('tensorboard --logdir=' + path + str(test_count))
    return


def calculate_win_percentage(scores):
    """calculates win percentage of each player based on scores array

    Args:
        scores (np.ndarray(shape=(games_to_report, len(player)): array of scores for last x games

    Returns:
        dict: player index as keys and win percentage as values
    """
    player_indices, win_counts = np.unique(np.argmax(scores,axis=1), return_counts=True)
    win_dict = dict(zip(player_indices, win_counts))
    for player_index in range(0, scores.shape[1]):
        if player_index not in win_dict:
            # if player has not won any games, add to dict with 0
            win_dict[player_index] = 0
        else:
            win_dict[player_index] /= len(scores)

    return win_dict


report_after_games = 100
games = 20000
players = [TensorforcePPOAgent1L(), TensorforcePPOAgent3L(),
           RuleBasedAgent(), AverageRandomPlayer()]
# seed(2)

### Tensorboard / Logging Stuff start
test_count = 1
path = 'logs/' + players[0].name + '/test_'
# Create a new path for logging the evaluation data
while os.path.exists(path + str(test_count)):
    test_count+=1
os.makedirs(path + str(test_count))

sess = tf.InteractiveSession()

# Create a new log file for every player in the selected path
file_writer_p1 = tf.summary.FileWriter(path + str(test_count) + '/' + players[0].__class__.__name__ + '_p1', sess.graph)
file_writer_p2 = tf.summary.FileWriter(path + str(test_count) + '/' + players[1].__class__.__name__ + '_p2', sess.graph)
file_writer_p3 = tf.summary.FileWriter(path + str(test_count) + '/' + players[2].__class__.__name__ + '_p3', sess.graph)
file_writer_p4 = tf.summary.FileWriter(path + str(test_count) + '/' + players[3].__class__.__name__ + '_p4', sess.graph)
file_writers = [file_writer_p1, file_writer_p2, file_writer_p3, file_writer_p4]

tf.global_variables_initializer().run()

# Run the tensorboard in a separate thread / alternativley comment this an run tensorboard in seperate terminal
t = threading.Thread(target=launch_tensor_board, args=([]))
t.start()
### Tensorboard Stuff end ###

# Runs the Game and Reports to Tensorboard
scores = np.empty(shape=(games, len(players)))
for i in range(games):
    # print("{}/{}".format(i, games))

    # play game
    wiz = Game(players=players)
    scores[i] = wiz.play_game()

    # report to tensorboard after every x games
    if i % report_after_games == 0 and i > 0:
        average_score = np.mean(scores[i - report_after_games:i], axis=0)
        win_percentage = calculate_win_percentage(scores[i - report_after_games:i])
        # loop through players and write values to disk for tensorboard use
        for index, player in enumerate(players):
            summary = tf.Summary()
            summary.value.add(tag="C) Score", simple_value=average_score[index])
            summary.value.add(tag="A) Win Percentage", simple_value=win_percentage[index])
            # ToDo change calculation of valid rate for RL Agent s.t. all played hands are considered and not last 10000
            if isinstance(players[index], RLAgent):
                summary.value.add(tag="B) Valid Rate", simple_value=players[index].valid_rate)
                summary.value.add(tag="E) Predictor Loss", simple_value=players[index].predictor.current_loss)
                summary.value.add(tag="D) Predictor Acc", simple_value=players[index].predictor.current_acc)

            file_writers[index].add_summary(summary, i)
            file_writers[index].flush()

# ToDo save model of best performing RL Agent --> based on valid rate or other metric ?
players[0].save_models()
print("Done")

