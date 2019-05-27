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
from agents.tensorforce.algorithms import TensorforcePPOAgent1L
from game_engine.player import AverageRandomPlayer
from agents.rule_based_agent import RuleBasedAgent

sess = tf.InteractiveSession()

test_count = 1

players = [TensorforcePPOAgent1L(), TensorforcePPOAgent1L(),
           RuleBasedAgent(), AverageRandomPlayer()]

path = 'logs/' + players[0].name + '/test_'
# Create a new path for logging the evaluation data
while os.path.exists(path + str(test_count)):
    test_count+=1
os.makedirs(path + str(test_count))

# Create a new log file in the selected path
file_writer_p1 = tf.summary.FileWriter(path + str(test_count) + '/' + players[0].__class__.__name__ + '_p1', sess.graph)
file_writer_p2 = tf.summary.FileWriter(path + str(test_count) + '/' + players[1].__class__.__name__ + '_p2', sess.graph)
file_writer_p3 = tf.summary.FileWriter(path + str(test_count) + '/' + players[2].__class__.__name__ + '_p3', sess.graph)
file_writer_p4 = tf.summary.FileWriter(path + str(test_count) + '/' + players[3].__class__.__name__ + '_p4', sess.graph)

file_writers = [file_writer_p1, file_writer_p2, file_writer_p3, file_writer_p4]

tf.global_variables_initializer().run()


def launchTensorBoard():
    os.system('tensorboard --logdir=' + path + str(test_count))
    return

# Run the tensorboard in a separate thread
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

games = 2000
# seed(2)

scores = []
for i in range(games):
    # print("{}/{}".format(i, games))
    wiz = Game(players=players)
    score = wiz.play_game()
    #if i % 100 == 0:
    # loop through players and write values to disk for tensorboard use
    # ToDo do this only every x games, calculate average for scores
    for index, player in enumerate(players):
        summary = tf.Summary()
        summary.value.add(tag="Score", simple_value=score[index])
        # ToDo add valid rate calculation to rule based player
        # ToDo change calculation of valid rate for RL Agent s.t. all played hands are considered and not last 10000
        if isinstance(players[index], RLAgent):
           summary.value.add(tag="Valid Rate", simple_value=players[index].valid_rate)

        file_writers[index].add_summary(summary, i)
        file_writers[index].flush()
    scores.append(score)
# ToDo save model of best performing RL Agent
players[0].save_models()
scores = np.array(scores)
print("Done")

