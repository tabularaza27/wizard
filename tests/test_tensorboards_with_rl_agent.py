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

sess = tf.InteractiveSession()

test_count = 1

players = [TensorforcePPOAgent1L(), AverageRandomPlayer(),
    AverageRandomPlayer(), AverageRandomPlayer()]

path = 'logs/' + players[0].name + '/test_'

# Create a new path for logging the evaluation data
while os.path.exists(path + str(test_count)):
    test_count+=1
os.makedirs(path + str(test_count))

# Create a new log file in the selected path
file_writer = tf.summary.FileWriter(path + str(test_count), sess.graph)

tf.global_variables_initializer().run()

# Run the tensorboard in a separate thread
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

games = 2000
# seed(2)

scores = []
for i in range(games):
    if i % 100 == 0:
        print("{}/{}".format(i, games))
    wiz = Game(players=players)
    score = wiz.play_game()
    summary = tf.Summary()
    summary.value.add(tag="Score", simple_value = score[0])
    file_writer.add_summary(summary, i)
    scores.append(score)
players[0].save_models()
scores = np.array(scores)
print("Done")

def launchTensorBoard():
    os.system('tensorboard --logdir=' + path + str(test_count))
    return