# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.utils import weights_init


class StateEncoder(torch.nn.Module):
    """
    state representation, phi(world state):
    """
    def __init__(self, input_dim, nb_channels=64, output_dim=128):
        super(StateEncoder, self).__init__()
        self.input_dim = input_dim
        self.nb_channels = nb_channels
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(input_dim[0], nb_channels, 1, stride=1)
        self.fc1 = nn.Linear(nb_channels * input_dim[1] * input_dim[2], output_dim)    

        # init weights
        self.apply(weights_init)


    def forward(self, state):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1],
                                            self.input_dim[2])
        x = F.relu(self.conv1(state))
        x = x.view(-1, self.nb_channels * self.input_dim[1] * self.input_dim[2])
        x = F.relu(self.fc1(x))
        return x


class StateActionEncoder(torch.nn.Module):
    """
    encoder of (state, action) pair
    (state, action) -> state_action_embed
    """
    def __init__(self,
                 input_dim,
                 action_size,
                 nb_channels=64,
                 state_action_embed_dim=128):
        super(StateActionEncoder, self).__init__()

        self.input_dim = input_dim
        self.action_size = action_size
        self.conv_feat_dim = (input_dim[1] - 0) * (input_dim[2] - 0) * nb_channels
        self.state_action_embed_dim = state_action_embed_dim
        self.conv1 = nn.Conv2d(input_dim[0] + action_size, nb_channels, 1, stride=1)
        self.fc1 = nn.Linear((input_dim[1] - 0) * (input_dim[2] - 0) * nb_channels, state_action_embed_dim)
        self.fc2 = nn.Linear(state_action_embed_dim, state_action_embed_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                action_2d):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_action = torch.cat([state, action_2d], 1)
        x = F.relu(self.conv1(state_action))
        x = x.view(-1, self.conv_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StateActionRewardEncoder(torch.nn.Module):
    """
    encoder of (state, action, reward) tuple
    (state, action, reward) -> state_action_reward_embed
    """
    def __init__(self,
                 input_dim,
                 action_size,
                 goal_dim,
                 nb_channels=64,
                 state_action_reward_embed_dim=128):
        super(StateActionRewardEncoder, self).__init__()

        self.input_dim = input_dim
        self.action_size = action_size
        self.goal_dim = goal_dim
        self.conv_feat_dim = input_dim[1] * input_dim[2] * nb_channels
        self.state_action_reward_embed_dim = state_action_reward_embed_dim
        self.conv1 = nn.Conv2d(input_dim[0] + action_size + goal_dim, nb_channels, 1, stride=1)
        self.fc1 = nn.Linear(input_dim[1] * input_dim[2] * nb_channels, state_action_reward_embed_dim)
        self.fc2 = nn.Linear(state_action_reward_embed_dim, state_action_reward_embed_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                action_2d,
                reward_2d):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_action_reward = torch.cat([state, action_2d, reward_2d], 1)
        x = F.relu(self.conv1(state_action_reward))
        x = x.view(-1, self.conv_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StateRewardEncoder(torch.nn.Module):
    """
    encoder of (state, reward) tuple
    (state, reward) -> state_reward_embed
    """
    def __init__(self,
                 input_dim,
                 reward_dim,
                 nb_channels=64,
                 state_reward_embed_dim=128):
        super(StateRewardEncoder, self).__init__()

        self.input_dim = input_dim
        self.reward_dim = reward_dim
        self.conv_feat_dim = input_dim[1] * input_dim[2] * nb_channels
        self.state_reward_embed_dim = state_reward_embed_dim
        self.conv1 = nn.Conv2d(input_dim[0] + reward_dim, nb_channels, 1, stride=1)
        self.fc1 = nn.Linear(input_dim[1] * input_dim[2] * nb_channels, state_reward_embed_dim)
        self.fc2 = nn.Linear(state_reward_embed_dim, state_reward_embed_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                reward_2d):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_reward = torch.cat([state, reward_2d], 1)
        x = F.relu(self.conv1(state_reward))
        x = x.view(-1, self.conv_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StateHistEncoder(torch.nn.Module):
    """
    encoder of (state, hist) pair
    (state, hist) -> state_stats_embed
    """
    def __init__(self,
                 input_dim,
                 hist_dim,
                 nb_channels=64,
                 state_hist_embed_dim=128):
        super(StateHistEncoder, self).__init__()

        self.input_dim = input_dim
        self.hist_dim = hist_dim
        self.conv_feat_dim = (input_dim[1] - 0) * (input_dim[2] - 0) * nb_channels
        self.state_hist_embed_dim = state_hist_embed_dim
        self.conv1 = nn.Conv2d(input_dim[0] + hist_dim, nb_channels, 1, stride=1)
        self.fc1 = nn.Linear((input_dim[1] - 0) * (input_dim[2] - 0) * nb_channels, state_hist_embed_dim)
        self.fc2 = nn.Linear(state_hist_embed_dim, state_hist_embed_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                hist_2d):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_hist= torch.cat([state, hist_2d], 1)
        x = F.relu(self.conv1(state_hist))
        x = x.view(-1, self.conv_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StateHistActionRewardEncoder(torch.nn.Module):
    """
    encoder of (state, hist, action, reward) tuple
    (state, hist, action, reward) -> state_hist_action_reward_embed
    """
    def __init__(self,
                 input_dim,
                 hist_dim,
                 action_size,
                 goal_dim,
                 nb_channels=64,
                 state_hist_action_reward_embed_dim=128):
        super(StateHistActionRewardEncoder, self).__init__()

        self.input_dim = input_dim
        self.hist_dim = hist_dim
        self.action_size = action_size
        self.goal_dim = goal_dim
        self.conv_feat_dim = input_dim[1] * input_dim[2] * nb_channels
        self.state_hist_action_reward_embed_dim = state_hist_action_reward_embed_dim
        self.conv1 = nn.Conv2d(input_dim[0] + hist_dim + action_size + goal_dim, nb_channels, 1, stride=1)
        self.fc1 = nn.Linear(input_dim[1] * input_dim[2] * nb_channels, state_hist_action_reward_embed_dim)
        self.fc2 = nn.Linear(state_hist_action_reward_embed_dim, state_hist_action_reward_embed_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                hist_2d,
                action_2d,
                reward_2d):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_hist_action_reward = torch.cat([state, hist_2d, action_2d, reward_2d], 1)
        x = F.relu(self.conv1(state_hist_action_reward))
        x = x.view(-1, self.conv_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
