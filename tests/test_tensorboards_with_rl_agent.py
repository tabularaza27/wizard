#!/usr/bin/env python3

"""Play specified number of games to train RL Agent"""

from random import seed
import numpy as np
# This allows the file to be run in a test folder as opposed to having to be in the root directory
import sys
sys.path.append('../')

import tensorflow as tf
from game_engine.game import Game
from agents.tensorforce.algorithms import TensorforcePPOAgent1L
from game_engine.player import AverageRandomPlayer
import tensorboard as tb
import tensorboard.program
import tensorboard.default

sess = tf.InteractiveSession()

file_writer = tf.summary.FileWriter('logs/', sess.graph)

tf.global_variables_initializer().run()

# tb.program.FLAGS.logdir = '/logs'
# tb.program.main(tb.default.get_plugins(),
#                 tb.default.get_assets_zip_provider())

games = 2000
# seed(2)
players = [TensorforcePPOAgent1L(), AverageRandomPlayer(),
    AverageRandomPlayer(), AverageRandomPlayer()]
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
