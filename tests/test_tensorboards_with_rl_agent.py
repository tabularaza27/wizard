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
from agents.tf_agents.tf_agents_ppo_agent import TFAgentsPPOAgent
from game_engine.player import AverageRandomPlayer
from agents.rule_based_agent import RuleBasedAgent

tf.compat.v1.enable_v2_behavior()


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
    player_indices, win_counts = np.unique(np.argmax(scores, axis=1), return_counts=True)
    win_dict = dict(zip(player_indices, win_counts))
    for player_index in range(0, scores.shape[1]):
        if player_index not in win_dict:
            # if player has not won any games, add to dict with 0
            win_dict[player_index] = 0
        else:
            win_dict[player_index] /= len(scores)

    return win_dict


self_play = True
report_after_games = 100
games = 2000000
rl_agent = TFAgentsPPOAgent()

if self_play:
    players = [rl_agent, rl_agent.clone(), rl_agent.clone(), rl_agent.clone()]
else:
    # use 2 rl agents, 1 normal rule_based agent, 1 rule based agent with predictor

    rule_based_agent = RuleBasedAgent()
    rule_based_agent_with_predictor = RuleBasedAgent(use_predictor=True)
    players = [rl_agent, rl_agent.clone(), rule_based_agent, rule_based_agent_with_predictor]

# seed(2)

### Tensorboard / Logging Stuff start
test_count = 1
path = 'logs/' + players[0].name + '/test_'
# Create a new path for logging the evaluation data
while os.path.exists(path + str(test_count)):
    test_count += 1
os.makedirs(path + str(test_count))

sess = tf.InteractiveSession()

# Create a new log file for every player in the selected path
file_writer_p1 = tf.contrib.summary.create_file_writer(
    path + str(test_count) + '/' + players[0].name + '_p1')
file_writer_p2 = tf.contrib.summary.create_file_writer(
    path + str(test_count) + '/' + players[1].name + '_p2')
file_writer_p3 = tf.contrib.summary.create_file_writer(
    path + str(test_count) + '/' + players[2].name + '_p3')
file_writer_p4 = tf.contrib.summary.create_file_writer(
    path + str(test_count) + '/' + players[3].name + '_p4')
file_writers = [file_writer_p1, file_writer_p2, file_writer_p3, file_writer_p4]

# Run the tensorboard in a separate thread / alternativley comment this an run tensorboard in seperate terminal
# t = threading.Thread(target=launch_tensor_board, args=([]))
# t.start()
### Tensorboard Stuff end ###

# Runs the Game and Reports to Tensorboard
scores = np.empty(shape=(games, len(players)))
for i in range(games):
    print("{}/{}".format(i, games))

    # play game
    wiz = Game(players=players)
    scores[i] = wiz.play_game()

    # report to tensorboard after every x games
    if i % report_after_games == 0 and i > 0:
        average_score = np.mean(scores[i - report_after_games:i], axis=0)
        win_percentage = calculate_win_percentage(scores[i - report_after_games:i])
        # loop through players and write values to disk for tensorboard use
        for index, player in enumerate(players):
            with file_writers[index].as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("2_score", average_score[index], step=i)
                tf.contrib.summary.scalar("1_win_percentage", win_percentage[index], step=i)
                if hasattr(players[index], 'predictor'):
                    tf.contrib.summary.scalar("4_predictor_loss", players[index].predictor.current_loss, step=i)
                    tf.contrib.summary.scalar("3_predictor_acc", players[index].predictor.current_acc, step=i)
                    # prediction difference
                    tf.contrib.summary.scalar("5_prediction_differences", np.mean(players[index].predictor.prediction_differences), step=i)
                    tf.contrib.summary.histogram("dist_prediction_differences", players[index].predictor.prediction_differences, step=i)
                    players[index].predictor.prediction_differences = []
                    # add distribution / mean of predictions for all rounds played and for every round (# cards)
                    for amount_cards in range(0, 16):
                        tf.contrib.summary.scalar("_{}_mean_prediction".format(amount_cards), np.mean(players[index].predictor.predictions[amount_cards]))
                        tf.contrib.summary.histogram("_{}_predictions".format(amount_cards), players[index].predictor.predictions[amount_cards], step=i)
                        players[index].predictor.predictions[amount_cards] = []
                if isinstance(players[index], RLAgent):
                    tf.contrib.summary.scalar("5_valid_rate", players[index].valid_rate, step=i)
        rl_agent.save_models(i)
        if not self_play:
            rule_based_agent_with_predictor.save_models()
