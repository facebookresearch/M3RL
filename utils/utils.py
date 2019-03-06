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
import sys
import random

import torch
from torch import nn
from torch.autograd import Variable


def update_network(args, hist_model, ind_model, global_model, loss, optimizer):
    """update network parameters"""
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(list(hist_model.parameters()) + list(ind_model.parameters()) + list(global_model.parameters()), 
                            args.max_gradient_norm, 1)
    optimizer.step()


def weights_init(m):
    """initializing weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.normal_(0, 0.01)


def normalized_p(p, blocked_actions = None):
    """normalize a distribution"""
    if blocked_actions:
        p[0, blocked_actions] = 0.0
    return p / sum(p[0])


def one_hot(value, dim, type='float'):
    """value to one hot vec (dim,)"""
    if type == 'float':
        one_hot = torch.FloatTensor(dim).zero_()
    else:
        one_hot = torch.LongTensor(dim).zero_()
    return one_hot.scatter_(-1, torch.LongTensor([value]), 1)


def one_hot_np(value, dim, type='float'):
    """value to one hot numpy vector"""
    if type == 'float':
        one_hot = np.zeros((dim, )).astype('float')
    else:
        one_hot = np.zeros((dim, )).astype('int')
    one_hot[value] = 1
    return one_hot


def expand(vec_tensor, output_dim):
    """expand vec tensor spatially"""
    return vec_tensor.repeat(1, output_dim[1] * output_dim[2]) \
                     .view(output_dim[1], output_dim[2], output_dim[0]) \
                     .permute(2, 0, 1) \
                     .unsqueeze(0)


def to_Variable(tensor, cuda):
    """convert a tensor to a variable"""
    return Variable(tensor.cuda()) if cuda else Variable(tensor)


def array2vec(array):
    """convert array to vector"""
    nb_dim = len(array.shape)
    vec_dim = 1
    for dim_id in range(nb_dim):
        vec_dim *= array.shape[dim_id]
    return array.reshape((vec_dim))


def normalize_among_agents(rewards):
    """normalize rewards so that sum_i r_i(g) = 1 for all g"""
    nb_agents = len(rewards)
    nb_goals = len(rewards[0])
    for goal in range(nb_goals):
        tot_reward = sum([reward[goal] for reward in rewards])
        if tot_reward > 0:
            for agent_id in range(nb_agents):
                rewards[agent_id][goal] /= tot_reward
    return rewards


def goals2rewards(goals, nb_goals):
    """convert goal to reward"""
    nb_agents = len(goals)
    rewards = [None] * nb_agents
    for agent_id, goal in enumerate(goals):
        rewards[agent_id] = [0] * nb_goals
        rewards[agent_id][goal] = 1
    return rewards


def rewards_from_goals_resources(goals, resources, nb_goals, nb_resources, selection=None):
    """convert goals and resources to rewards"""
    nb_agents = len(goals) if selection is None else len(selection)
    rewards = [[0] * (nb_goals * nb_resources) for _ in range(nb_agents)]
    if selection is None:
        for agent_id, goal, resource in zip(range(nb_agents), goals, resources):
            rewards[agent_id][goal * nb_resources + resource] = 1
    else:
        cnt_selected_agents = 0
        for agent_id, sel in enumerate(selection):
            if sel:
                goal, resource = goals[cnt_selected_agents], resources[cnt_selected_agents]
                rewards[agent_id][goal * nb_resources + resource] = 1
                cnt_selected_agents += 1
    return rewards


def rewards_from_goals_payments(goals, payments, nb_goals, nb_pay_types, selection=None):
    """convert goals and payments to rewards"""
    nb_agents = len(goals) if selection is None else len(selection)
    rewards = [[0] * (nb_goals * nb_pay_types) for _ in range(nb_agents)]
    if selection is None:
        for agent_id, goal, pay in zip(range(nb_agents), goals, payments):
            rewards[agent_id][goal * nb_pay_types + pay] = 1
    else:
        cnt_selected_agents = 0
        for agent_id, sel in enumerate(selection):
            if sel:
                goal, pay = goals[cnt_selected_agents], payments[cnt_selected_agents]
                rewards[agent_id][goal * nb_pay_types + pay] = 1
                cnt_selected_agents += 1
    return rewards


def weighted_sum(values, weights):
    """compute weighted sum of a list of values"""
    return sum([value * weight for value, weight in zip(values, weights)])


def append_list_running_average(cur_list, new_value, update_rate=0.05):
    """appending running average to a list"""
    if cur_list:
        return cur_list + [cur_list[-1] * (1 - update_rate) + new_value * update_rate]
    else:
        return [new_value]
