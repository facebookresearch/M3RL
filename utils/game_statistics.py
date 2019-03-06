# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import copy
import math

import torch
from torch.autograd import Variable


class WorkerStats():
    """game stats of individual worker agents
    with configurable window size (1 by default)
    """
    def __init__(self, nb_goal_types, nb_pay_types, window_size, max_episode_length, update_rate=0.05):
        self.nb_goal_types = nb_goal_types
        self.nb_pay_types = nb_pay_types
        self.window_size = window_size
        self.max_episode_length = max_episode_length
        self.nb_windows = max_episode_length // window_size
        self.nb_commits = [[0] * self.nb_windows for _ in range(nb_goal_types)]
        self.ave_success = [[0] * self.nb_windows for _ in range(nb_goal_types)]
        self.nb_proposals = [[0] * nb_pay_types for _ in range(nb_goal_types)]
        self.ave_commit = [[0] * nb_pay_types for _ in range(nb_goal_types)]
        self.update_rate = update_rate
        self.stats_dim = nb_goal_types * (nb_pay_types + self.nb_windows)
        self.commit_offset = nb_goal_types * self.nb_windows
        self.feat = np.zeros((1, self.stats_dim))
        self.nb_episodes = 0


    def set_stats(self, nb_commits, ave_success, nb_proposals, ave_commit, feat, nb_episodes):
        """set the stats"""
        self.nb_commits = copy.deepcopy(nb_commits)
        self.ave_success = copy.deepcopy(ave_success)
        self.nb_proposals = copy.deepcopy(nb_proposals)
        self.ave_commit = copy.deepcopy(ave_commit)
        self.feat = feat
        self.nb_episodes = nb_episodes


    def reset_stats(self):
        self.nb_commits = [[0] * self.nb_windows for _ in range(self.nb_goal_types)]
        self.ave_success = [[0] * self.nb_windows for _ in range(self.nb_goal_types)]
        self.nb_proposals = [[0] * self.nb_pay_types for _ in range(self.nb_goal_types)]
        self.ave_commit = [[0] * self.nb_pay_types for _ in range(self.nb_goal_types)]
        self.feat = np.zeros((1, self.stats_dim))
        self.nb_episodes = 0


    def add_episode(self):
        self.nb_episodes += 1


    def update_success(self, goal, steps, success):
        """update success rate since commitment"""
        t = min(steps - 1, self.max_episode_length - 1)
        window_id = t // self.window_size
        window_id = min(max(window_id, 0), self.nb_windows - 1)
        self.ave_success[goal][window_id] = (1 - self.update_rate) * self.ave_success[goal][window_id] + self.update_rate * success
        self.nb_commits[goal][window_id] += 1
        self.feat[0, goal * self.nb_windows + window_id] = self.ave_success[goal][window_id]


    def update_commit(self, goal, pay, commit):
        """update commitment rate"""
        self.ave_commit[goal][pay] = (1 - self.update_rate) * self.ave_commit[goal][pay] + self.update_rate * commit
        self.nb_proposals[goal][pay] += 1
        self.feat[0, self.commit_offset + goal * self.nb_pay_types + pay] = self.ave_commit[goal][pay]


    def get_stats_feat(self):
        """get stats feature"""
        return self.feat


    def get_stats_var(self, cuda):
        """get stats as a Variable"""
        return Variable(torch.from_numpy(self.feat).float().cuda()) if cuda else Variable(torch.from_numpy(self.feat).float())


    def get_prob_commits(self):
        """get exploration probilities"""
        prob = np.ones((1, self.nb_goal_types))
        return prob


    def get_prob_proposals(self):
        """get exploration probilities"""
        prob = np.ones((1, self.nb_goal_types))
        return prob
