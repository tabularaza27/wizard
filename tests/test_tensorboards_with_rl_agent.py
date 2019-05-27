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
    # if i % 100 == 0:
    #     print("{}/{}".format(i, games))
    wiz = Game(players=players)
    score = wiz.play_game()
    summary = tf.Summary()
    summary.value.add(tag="Score", simple_value = score[0])
    # summary.value.add(tag="Valid Rate", simple_value = players[0].valid_rate)
    file_writer_p1.add_summary(summary, i)
    file_writer_p1.flush()
    summary = tf.Summary()
    summary.value.add(tag="Score", simple_value=score[1])
    # summary.value.add(tag="Valid Rate", simple_value=players[1].valid_rate)
    file_writer_p2.add_summary(summary, i)
    file_writer_p2.flush()
    summary = tf.Summary()
    summary.value.add(tag="Score", simple_value=score[2])
    # summary.value.add(tag="Valid Rate", simple_value=players[2].valid_rate)
    file_writer_p3.add_summary(summary, i)
    file_writer_p3.flush()
    summary = tf.Summary()
    summary.value.add(tag="Score", simple_value=score[3])
    # summary.value.add(tag="Valid Rate", simple_value=players[3].valid_rate)
    file_writer_p4.add_summary(summary, i)
    file_writer_p4.flush()
    scores.append(score)
players[0].save_models()
scores = np.array(scores)
print("Done")

