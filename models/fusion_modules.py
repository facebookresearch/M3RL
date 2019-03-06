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


class FusionCat(torch.nn.Module):
    """
    Fusion by concat
    """
    def __init__(self, input_dim, latent_dim, normalized=False):
        super(FusionCat, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fusion = nn.Linear(latent_dim * 2, self.output_dim)

        self.apply(weights_init)

    
    def forward(self, x, y):
        y_embed = F.relu(self.fc1(y))
        x_y_cat = torch.cat([x, y_embed.view(-1, self.latent_dim)], -1)
        x_y_fusion = self.fusion(x_y_cat)
        return x_y_fusion


class FusionAtt(torch.nn.Module):
    """
    Fusion by attention
    """
    def __init__(self, input_dim, output_dim, normalized=False):
        super(FusionAtt, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalized = normalized
        
        self.att_linear = nn.Linear(input_dim, output_dim)
    
        self.apply(weights_init)
    
    
    def forward(self, x, y):
        if self.normalized:
            att = F.sigmoid(self.att_linear(y))
        else:
            att = self.att_linear(y)
        x_y_fusion = x * att
        return x_y_fusion


class StateMindEncoder(torch.nn.Module):
    """
    encoder of (state, mind) pair
    (state, mind) -> state_mind_embed
    """
    def __init__(self,
                 input_dim,
                 mind_dim,
                 latent_dim,
                 nb_channels=32):
        super(StateMindEncoder, self).__init__()

        self.input_dim = input_dim
        self.mind_dim = mind_dim
        self.state_mind_embed_dim = nb_channels * input_dim[1] * input_dim[2]
        self.nb_channels = nb_channels
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(input_dim[0], nb_channels, 1, stride=1)
        self.att_linear = nn.Linear(mind_dim, nb_channels)
        self.fc1 = nn.Linear(self.state_mind_embed_dim, latent_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                mind):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        x = F.relu(self.conv1(state))
        att = F.sigmoid(self.att_linear(mind))
        x *= att.unsqueeze(-1).unsqueeze(-1)
        x = x.view(-1, self.state_mind_embed_dim)
        x = self.fc1(x)
        return x


class StateRewardMindEncoder(torch.nn.Module):
    """
    encoder of (state, reward, mind) tuple
    (state, reward, mind) -> state_mind_embed
    """
    def __init__(self,
                 input_dim,
                 reward_dim,
                 mind_dim,
                 latent_dim,
                 nb_channels=32):
        super(StateRewardMindEncoder, self).__init__()

        self.input_dim = input_dim
        self.reward_dim = reward_dim
        self.mind_dim = mind_dim
        self.state_mind_embed_dim = nb_channels * input_dim[1] * input_dim[2]
        self.nb_channels = nb_channels
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(input_dim[0] + reward_dim, nb_channels, 1, stride=1)
        self.att_linear = nn.Linear(mind_dim, nb_channels)
        self.fc1 = nn.Linear(self.state_mind_embed_dim, latent_dim)

        # init weights
        self.apply(weights_init)

    
    def forward(self, 
                state,
                received_reward_2d,
                mind):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        state_reward = torch.cat([state, received_reward_2d], 1)

        x = F.relu(self.conv1(state_reward))
        att = self.att_linear(mind)
        x *= att.unsqueeze(-1).unsqueeze(-1)
        x = x.view(-1, self.state_mind_embed_dim)
        x = F.relu(self.fc1(x))
        return x


class Context(torch.nn.Module):
    """
    Combine individual latent vectors
    """
    def __init__(self, input_dim, output_dim, pooling='sum'):
        super(Context, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim    
        self.pooling = pooling
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        
        self.apply(weights_init)

    
    def forward(self, feat_list, hidden=None):
        if self.pooling == 'sum':
            x = sum(feat_list)
        elif self.pooling == 'average':
            x = sum(feat_list) / len(feat_list)
        else:
            raise ValueError('invalid pooling method')
            return None
        c = F.relu(self.fc1(x))
        return c
