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

from .state_modules import *
from .fusion_modules import *
from .mind_modules import *
from utils.utils import weights_init


class PolicyNet2(torch.nn.Module):
    """
    policy net for both policies
    """
    def __init__(self, input_dim, latent_dim, goal_dim, resource_dim, activation='softmax'):
        super(PolicyNet2, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.resource_dim = resource_dim
        self.activation = activation

        self.fc = nn.Linear(input_dim, latent_dim)
        self.actor_goal = nn.Linear(latent_dim, goal_dim)
        self.actor_resource = nn.Linear(latent_dim, resource_dim)

        self.apply(weights_init)


    def forward(self, x):
        x = F.relu(self.fc(x))
        x_goal = self.actor_goal(x)
        policy_goal = F.softmax(x_goal, dim=-1)
        x_resource = self.actor_resource(x)
        policy_resource = F.softmax(x_resource, dim=-1)
        return policy_goal, policy_resource


class HistEncoder(torch.nn.Module):
    """
    Encoding an agent's history
    """
    def __init__(self, state_dim, reward_dim, action_size,
                 nb_channels=64, latent_dim=128, hist_dim=128, lstm=True):
        super(HistEncoder, self).__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.action_size = action_size
        self.nb_channels = nb_channels
        self.latent_dim = latent_dim
        self.hist_dim = hist_dim
        self.lstm = lstm

        self.state_action_reward = StateActionRewardEncoder(state_dim, action_size, reward_dim, nb_channels, latent_dim)
        self.mind_tracker = MindTracker(latent_dim, hist_dim, lstm)


    def forward(self, received_reward_2d, cur_state, action_2d, hidden_state=None):
        cur_state_action_reward_feat = self.state_action_reward(cur_state, action_2d, received_reward_2d)
        hist_mind, hidden_state = self.mind_tracker(cur_state_action_reward_feat, hidden_state)

        return hist_mind, hidden_state


class WorkerStatsEncoder(torch.nn.Module):
    """
    Encoding worker agent stats
    """
    def __init__(self, stats_dim, hist_dim=128):
        super(WorkerStatsEncoder, self).__init__()
        self.stats_dim = stats_dim
        self.hist_dim = hist_dim

        self.fc = nn.Linear(stats_dim, hist_dim)


    def forward(self, stats):
        x = F.relu(self.fc(stats))
        return x


class MindPolicyPredHist(torch.nn.Module):
    """
    individual agent policy prediction based on mind modeling and history
    """
    def __init__(self, state_dim, reward_dim, action_size, goal_dim,
                 nb_channels=64, latent_dim=128, mind_dim=128, hist_dim=128, lstm=True,
                 fusion='att', att_norm=False, pooling='sum', activation='linear'):
        super(MindPolicyPredHist, self).__init__()
        self.state_dim = state_dim        
        self.reward_dim = reward_dim
        self.action_size = action_size
        self.goal_dim = goal_dim
        self.nb_channels = nb_channels
        self.latent_dim = latent_dim
        self.mind_dim = mind_dim
        self.lstm = lstm
        self.fusion = fusion
        self.att_norm = att_norm
        self.pooling = pooling
        self.activation = activation

        self.state = StateEncoder(state_dim, nb_channels, latent_dim)
        self.state_action_reward = StateActionRewardEncoder(state_dim, action_size, reward_dim, nb_channels, latent_dim)
        self.mind_tracker = MindTracker(latent_dim, mind_dim, lstm)
        self.mind_hist_cat_fusion = nn.Linear(mind_dim + hist_dim, mind_dim)
        self.policy_pred = nn.Linear(mind_dim, action_size)


    def forward(self, prev_action_2d, received_reward_2d, cur_state, hist_mind, hidden_state=None):
        cur_state_action_reward_feat = self.state_action_reward(cur_state, prev_action_2d, received_reward_2d)
        mind, hidden_state = self.mind_tracker(cur_state_action_reward_feat, hidden_state)
        fused_mind = F.relu(self.mind_hist_cat_fusion(torch.cat([hist_mind, mind], -1)))
        pred_policy = F.log_softmax(self.policy_pred(fused_mind), -1)
        return mind, pred_policy, hidden_state


class Critic(torch.nn.Module):
    """centralized critic"""
    def __init__(self, context_dim=128):
        super(Critic, self).__init__()
        self.context_dim = context_dim
        self.critic_linear = nn.Linear(context_dim, 1)

        self.apply(weights_init)

    def forward(self, context):
        value = self.critic_linear(context)
        return value


class CriticVGVC(torch.nn.Module):
    """centralized critic V_G & V_C"""
    def __init__(self, context_dim, goal_dim, pay_dim):
        super(CriticVGVC, self).__init__()
        self.context_dim = context_dim
        self.goal_dim = goal_dim
        self.pay_dim = pay_dim

        self.critic_linear = nn.Linear(context_dim, goal_dim)
        self.critic_pay_linear = nn.Linear(context_dim, pay_dim)

        self.apply(weights_init)

    def forward(self, context):
        value = self.critic_linear(context)
        value_pay = self.critic_pay_linear(context)
        return value, value_pay


class ActorCriticFusionVGVCHist(torch.nn.Module):
    """full model"""
    def __init__(self, state_dim, goal_dim, resource_dim,
                 latent_dim=128, mind_dim=128, hist_dim=128, nb_channels=32,
                 fusion='att', att_norm=False, pooling='sum',
                 policy_activation='softmax'):
        super(ActorCriticFusionVGVCHist, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.resource_dim = resource_dim
        self.latent_dim = latent_dim
        self.mind_dim = mind_dim
        self.hist_dim = hist_dim
        self.nb_channels = nb_channels
        self.fusion = fusion
        self.att_norm = att_norm
        self.pooling = pooling
        self.policy_activation=policy_activation

        if fusion == 'att':
            self.state_encoder = StateEncoder(state_dim, nb_channels, latent_dim)
            self.state_mind_fusion = FusionAtt(mind_dim + hist_dim, latent_dim, normalized=att_norm)
        else:
            self.state_mind_fusion = StateMindEncoder(state_dim, mind_dim + hist_dim, latent_dim, nb_channels)
        self.mind_context = Context(latent_dim, latent_dim, pooling)
        self.actor = PolicyNet2(latent_dim * 2, latent_dim, goal_dim, resource_dim, policy_activation)
        self.critic = CriticVGVC(latent_dim, goal_dim, resource_dim)


    def forward(self, cur_states, minds, hist_minds):
        if self.fusion == 'att':
            state_minds = [self.state_mind_fusion(self.state_encoder(state), torch.cat([mind, hist_mind], -1)) 
                            for state, mind, hist_mind in zip(cur_states, minds, hist_minds)]
        else:
            state_minds = [self.state_mind_fusion(state, mind) for state, mind in zip(cur_states, minds)]
        context_all = self.mind_context(state_minds)
        context_state_minds = torch.cat([torch.cat([context_all, state_mind], -1) for state_mind in state_minds])
        policies_goal, policies_resource = self.actor(context_state_minds)
        values, values_pay = self.critic(context_all)
        return policies_goal, policies_resource, values, values_pay


class ActorCriticFusionHist(torch.nn.Module):
    """w/o SR"""
    def __init__(self, state_dim, goal_dim, resource_dim,
                 latent_dim=128, mind_dim=128, hist_dim=128, nb_channels=32, 
                 fusion='att', att_norm=False, pooling='sum',
                 policy_activation='softmax'):
        super(ActorCriticFusionHist, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.resource_dim = resource_dim
        self.latent_dim = latent_dim
        self.mind_dim = mind_dim
        self.hist_dim = hist_dim
        self.nb_channels = nb_channels
        self.fusion = fusion
        self.att_norm = att_norm
        self.pooling = pooling
        self.policy_activation=policy_activation

        if fusion == 'att':
            self.state_encoder = StateEncoder(state_dim, nb_channels, latent_dim)
            self.state_mind_fusion = FusionAtt(mind_dim + hist_dim, latent_dim, normalized=att_norm)
        else:
            self.state_mind_fusion = StateMindEncoder(state_dim, mind_dim + hist_dim, latent_dim, nb_channels)
        self.mind_context = Context(latent_dim, latent_dim, pooling)
        self.actor = PolicyNet2(latent_dim * 2, latent_dim, goal_dim, resource_dim, policy_activation)
        self.critic = Critic(latent_dim)


    def forward(self, cur_states, minds, hist_minds):
        if self.fusion == 'att':
            state_minds = [self.state_mind_fusion(self.state_encoder(state), torch.cat([mind, hist_mind], -1)) 
                            for state, mind, hist_mind in zip(cur_states, minds, hist_minds)]
        else:
            state_minds = [self.state_mind_fusion(state, mind) for state, mind in zip(cur_states, minds)]
        context_all = self.mind_context(state_minds)
        context_state_minds = torch.cat([torch.cat([context_all, state_mind], -1) for state_mind in state_minds])
        policies_goal, policies_resource = self.actor(context_state_minds, hidden_goal, hidden_resource)
        values = self.critic(context_all)
        return policies_goal, policies_resource, values

