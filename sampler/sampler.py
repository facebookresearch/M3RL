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
import random
import torch
from torch.autograd import Variable
from utils.utils import normalized_p


class EGreedy:
    """epsilon greedy"""
    def __init__(self, action_size, init_epsilon, final_epsilon, max_exp_steps):
        self.action_size = action_size
        self.init_epsilon = self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.max_exp_steps = max_exp_steps
        self.steps = 0


    def reset(self):
        self.epsilon = self.init_epsilon
        self.steps = 0


    def set_zero(self):
        self.epsilon = 0.0


    def update(self):
        self.steps += 1
        self.epsilon = (self.init_epsilon - self.final_epsilon) \
                        * (1.0 - float(self.steps) / float(self.max_exp_steps)) \
                        + self.final_epsilon

        
    def sample(self, policy, cuda, return_prob=False, blocked_actions=None):
        """sample action w.r.t. given policy"""
        u = random.uniform(0, 1)
        if u < self.epsilon:
            p = normalized_p(np.ones(policy.shape), blocked_actions)
        else:
            p = normalized_p(policy, blocked_actions)
        if cuda:
            new_policy = Variable(torch.from_numpy(p).float().cuda())
        else:
            new_policy = Variable(torch.from_numpy(p).float())
        action = new_policy.multinomial()
        if return_prob:
            return action, p
        else:
            return action


class ArgMax:
    """ArgMax"""
    def __init__(self, action_size):
        self.action_size = action_size

        
    def sample(self, policy, blocked_actions=None):
        """sample action w.r.t. given policy"""
        action = policy.argmax(axis = 1)
        return action
