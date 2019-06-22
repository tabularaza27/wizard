#!/usr/bin/env python3

# This allows the file to be run in a test folder
# as opposed to having to be in the root directory
import sys
sys.path.append('../')

import os
import datetime
import json
import random
import subprocess
import itertools
import atexit

import numpy as np
import tensorflow as tf

from game_engine.game import Game
from game_engine.player import AverageRandomPlayer
from agents.rl_agent import RLAgent
from agents.tf_agents.tf_agents_ppo_agent import TFAgentsPPOAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.featurizers import OriginalFeaturizer

class TensorboardWrapper:
    def __init__(self):
        self.create_logdir()
        self.launch_tensorboard()

    def create_logdir(self):
        path = 'logs/training/'

        test_count = 1
        while os.path.exists(path + str(test_count)):
            test_count += 1

        self.logdir = path + str(test_count) + '/'
        os.makedirs(self.logdir)

    def launch_tensorboard(self):
        tensorboard = subprocess.Popen(['tensorboard',
                '--logdir', self.logdir,
                '--reload_interval', str(10)],
            stdout=open(self.logdir + 'stdout', 'w'),
            stderr=open(self.logdir + 'stderr', 'w'))
        atexit.register(tensorboard.terminate)

    def set_game_num(self, game_num):
        self.game_num = game_num

    def view_as(self, agent, name=None):
        return TensorboardAgentView(self, agent, name)

class TensorboardAgentView:
    def __init__(self, tensorboard_wrapper, agent, name):
        self.tb = tensorboard_wrapper

        if name is None:
            name = agent.name
        self.filewriter = tf.contrib.summary.create_file_writer(self.tb.logdir + name)

    def scalar(self, name, value):
        with self.filewriter.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name, value, step=self.tb.game_num)

    def histogram(self, name, value):
        with self.filewriter.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.histogram(name, value, step=self.tb.game_num)

class AgentPool:
    def __init__(self, main_agent, path=None):
        self.pool = []
        self.agent = main_agent

        if path is None:
            path = 'pools/MainPool'
        self.path = path

        if os.path.exists(path):
            self._load()
        else:
            os.makedirs(os.path.dirname(path))

        self.precomputed_clones = [self.agent.clone() for p in range(3)]

    def select_players(self):
        if len(self.pool) < 3:
            return [self.agent] + self.precomputed_clones
        return [self.agent] + random.sample(self.pool, 3)

    def add_current_version(self):
        clone_name = self.agent.name + '@' + datetime.datetime.now().isoformat()

        clone = self.agent.clone(clone_name)
        clone.save_models()

        # we discard that clone and load it again because it currently shares models
        # with our original agent, i.e. once the original agent is updated,
        # this clone is also updated which we don't want

        clone = self.agent.__class__(name=clone_name,
            keep_models_fixed=True, featurizer=self.agent.featurizer)
        self.pool.append(clone)
        self.save()

    def save(self):
        with open(self.path, 'w') as f:
            json.dump([agent.name for agent in self.pool], f, indent=4)

    def _load(self):
        with open(self.path) as f:
            pool_data = json.load(f)
            for agent_name in pool_data:
                agent = self.agent.__class__(name=agent_name,
                    keep_models_fixed=True, featurizer=self.agent.featurizer)
                self.pool.append(agent)

def tensorboard_plot(agent, tb, avg_score, win_percentage):
    tb.scalar('1_win_percentage', win_percentage)
    tb.scalar('2_score', avg_score)

    if hasattr(agent, 'valid_rate'):
        tb.scalar('7_valid_rate', agent.valid_rate)

    if not hasattr(agent, 'predictor'):
        return

    # this value is only available after the predictor has been trained
    # the frequency of this may be different than plotting frequency
    if agent.predictor.current_loss is not None:
        tb.scalar('5_predictor_loss', agent.predictor.current_loss)
        tb.scalar('3_predictor_acc', agent.predictor.current_acc)

        agent.predictor.current_loss = None
        agent.predictor.current_acc = None

    prediction_differences = agent.predictor.prediction_differences
    if len(prediction_differences) > 0:
        tb.scalar('6_prediction_differences', np.mean(prediction_differences))
        tb.histogram('1_prediction_differences', prediction_differences)

        # real prediction accuracy
        prediction_accuracy = (len(prediction_differences)
            - np.count_nonzero(prediction_differences)) / len(prediction_differences)
        tb.scalar('4_predictor_acc_real', prediction_accuracy)
    agent.predictor.prediction_differences = []

    for amount_cards in range(0, 16):
        if len(agent.predictor.predictions['overall'][amount_cards]) == 0:
            continue

        # mean predictions
        for plt_name, datapoint_name in [
                ('7_overall_mean_predictions', 'overall'),
                ('8_correct_mean_predictions', 'correct_prediction'),
                ('9_incorrect_mean_predictions', 'incorrect_prediction')]:
            data = agent.predictor.predictions[datapoint_name][amount_cards]
            if len(data) == 0:
                continue
            tb.scalar(f'{plt_name}_{amount_cards}', np.mean(data))

        # prediction distributions
        for plt_name, datapoint_name in [
                ('2_overall_predictions', 'overall'),
                ('3_correct_predictions', 'correct_prediction'),
                ('4_incorrect_predictions', 'incorrect_prediction')]:
            tb.histogram('{plt_name}_{amount_cards}',
                agent.predictor.predictions[datapoint_name][amount_cards])

        # reset predictions variable
        for e in ['overall', 'correct_prediction', 'incorrect_prediction']:
            agent.predictor.predictions[e][amount_cards] = []

def calculate_win_percentage(scores):
    scores = np.array(scores)
    player_indices, win_counts = np.unique(np.argmax(scores, axis=1), return_counts=True)
    win_percentages = np.zeros(scores.shape[1])
    win_percentages[player_indices] = win_counts / len(scores)
    return win_percentages

def plot_agents(tb, scores, agents, agents_to_plot):
    agents_to_plot = [(p, agents[p], tb.view_as(agents[p])) for p in agents_to_plot]
    mean_scores = np.mean(scores, axis=0)
    win_percentages = calculate_win_percentage(scores)

    for agent_position, agent, agent_tb_view in agents_to_plot:
        tensorboard_plot(agent, agent_tb_view,
            mean_scores[agent_position], win_percentages[agent_position])

def play_games(player_selector, tb, agents_to_plot, flags):
    scores = []
    for game_num in itertools.count():
        print(game_num)
        agents = player_selector()
        scores.append(Game(players=agents).play_game())

        if game_num == 0:
            continue

        if game_num % flags['tensorboard_plot_frequency'] == 0:
            tb.set_game_num(game_num)
            plot_agents(tb, scores, agents, agents_to_plot)
            scores = []

        yield game_num

def train_with_self_play_against_newest_version(tb, flags):
    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer())
    agents = [agent, agent.clone(), agent.clone(), agent.clone()]

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()

def train_with_self_play_against_old_versions(tb, flags):
    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer())
    agent_pool = AgentPool(agent)

    for game_num in play_games(agent_pool.select_players, tb, [0], flags):
        if game_num % flags['agent_save_frequency'] == 0:
            agent.save_models()

        if game_num % flags['pool_save_frequency'] == 0:
            agent_pool.add_current_version()

def evaluate(tb, flags, other_agents):
    agent = TFAgentsPPOAgent(featurizer=OriginalFeaturizer(), keep_models_fixed=True)
    agents = [agent] + other_agents(agent)

    for game_num in play_games(lambda: agents, tb, range(4), flags):
        pass

def evaluate_rule_based(tb, flags):
    agents = [RuleBasedAgent()] + [AverageRandomPlayer(
        name='AverageRandomPlayer' + str(i)) for i in range(3)]
    for game_num in play_games(lambda: agents, tb, range(4), flags):
        pass

def main():
    default_flags = ({
        'tensorboard_plot_frequency': 20,
        'agent_save_frequency': 50,
        'pool_save_frequency': 100,
    })

    # TODO maybe also make it possible to specify these flags as command line options
    flags = default_flags

    subcmds = ({
        'train_vs_old_self': (train_with_self_play_against_old_versions, []),
        'train_vs_current_self': (train_with_self_play_against_newest_version, []),
        'evaluate': (evaluate, [lambda agent:
            [agent.clone(), RuleBasedAgent(use_predictor=True), RuleBasedAgent()]]),
        # TODO some other evaluate_something could also be added here
        # which uses other opponents
        # TODO we allow the rule based agent predictor to learn while evaluating against it
        # maybe it should learn stuff before and then be fixed ?
        # But on the other hand if we can, while we are fixed, beat an agent which is still
        # learning against us, that's also not bad
        'evaluate_rule_based': (evaluate_rule_based, [])
    })

    if len(sys.argv) > 1:
        selected_subcmd = sys.argv[1]
    else:
        selected_subcmd = 'train_vs_old_self'

    selected_fn, args = subcmds[selected_subcmd]
    selected_fn(TensorboardWrapper(), flags, *args)

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.InteractiveSession()

    main()
