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
from pathlib import Path
import sys
import random
import time
import math
import pickle

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from models.policy_value_modules import WorkerStatsEncoder, MindPolicyPredHist, ActorCriticFusionVGVCHist
from sampler import EGreedy, ArgMax
from utils.utils import *
from utils.game_statistics import *
from envs.resource_collection import *
from envs.crafting import *
from .worker_agents import *


np.set_printoptions(precision=2, suppress=True)


def _str2class(str):
    """call a class using a string"""
    return getattr(sys.modules[__name__], str)


class M3RL:
    """
    M^3RL: Mind-aware Mulit-agent Management Reinforcement Learning 
    """
    def __init__(self, args):
        self.args = args
        # random seed
        random.seed(args.seed)
        # specify environment
        self.env = _str2class(args.exp_name)(nb_agent_types=args.nb_agent_types,
                                             nb_resource_types=args.nb_resource_types,
                                             nb_pay_types=args.nb_pay_types,
                                             include_type=False,
                                             obstacle=args.obstacle)

        # build model
        self.hist_module = WorkerStatsEncoder(
                                        stats_dim=args.nb_resource_types * \
                                            (args.nb_pay_types + \
                                             args.max_episode_length // args.window_size), 
                                        hist_dim=args.hist_dim)

        self.mind_tracker = MindPolicyPredHist(
                                        state_dim=self.env.obs_dim, 
                                        reward_dim=self.env.nb_goal_types * self.env.nb_pay_types,
                                        action_size=self.env.action_size,
                                        goal_dim=self.env.nb_goal_types,
                                        nb_channels=args.nb_channels,
                                        latent_dim=args.latent_dim,
                                        mind_dim=args.mind_dim,
                                        hist_dim=args.hist_dim,
                                        lstm=args.lstm_mind,
                                        fusion=args.fusion,
                                        att_norm=args.att_norm,
                                        pooling=args.pooling,
                                        activation='linear')

        self.manager_module = ActorCriticFusionVGVCHist(
                                        state_dim=self.env.obs_dim,
                                        goal_dim=self.env.nb_goal_types,
                                        resource_dim=self.env.nb_pay_types,
                                        latent_dim=args.latent_dim,
                                        mind_dim=args.mind_dim,
                                        hist_dim=args.hist_dim,
                                        fusion=args.fusion,
                                        att_norm=args.att_norm,
                                        pooling=args.pooling,
                                        policy_activation=args.policy_activation)
        if self.args.cuda:
            self.hist_module.cuda()
            self.mind_tracker.cuda()
            self.manager_module.cuda()

        self.run_name = "seed{}_nbagents{}_nbres{}_nbpay{}_intvl{}_noise{}_il{}" \
                        .format(args.seed,
                                args.max_nb_agents,
                                args.max_nb_resources,
                                args.nb_pay_types,
                                args.interval,
                                args.noise,
                                int(args.IL))

        self.checkpoint_dir = "".join([args.checkpoint_dir, '/', args.exp_name, '/', self.run_name])
        p = Path(self.checkpoint_dir)
        if not p.is_dir():
            p.mkdir(parents = True)
        self.record_dir = "".join([args.record_dir, "/", args.exp_name, '/', self.run_name])
        p = Path(self.record_dir)
        if not p.is_dir():
            p.mkdir(parents = True)

        # for sampling bonus
        self.sampler_egreedy = EGreedy(self.env.nb_pay_types, 
                                       args.init_epsilon, 
                                       args.final_epsilon, 
                                       args.max_exp_steps)
        # for sampling goal
        self.sampler_egreedy0 = EGreedy(self.env.nb_resource_types, 
                                       0, 
                                       0, 
                                       args.max_exp_steps)
        # for sampling goal during agent-wise exploration (uniform samples)
        self.sampler_exp = EGreedy(self.env.nb_resource_types, 0, 0, args.max_exp_steps)
        # for predicting action
        self.sampler_argmax = ArgMax(self.env.action_size)
        self.worker_agents = [] # worker agents
        self.worker_stats = dict() # work agents' UCB style stats


    def encode_history_workers(self):
        """encode the recent history of all present agents in the game"""
        hist_mind_list = []
        for agent in self.env.agents:
            agent_identity = agent['identity']
            hist_mind = self.hist_module(self.worker_stats[agent_identity].get_stats_var(self.args.cuda))
            hist_mind_list.append(hist_mind)
        self.hist_minds = torch.cat(hist_mind_list)


    def select_actions_workers(self, goal_list, payment_list):
        """get ground-truth action based on ground-truth policy"""
        return [worker_agent.select_action(self.env, goal, payment) 
                    for worker_agent, goal, payment in zip(self.worker_agents, goal_list, payment_list)]


    def pred_actions_workers(self, model, prev_actions, cur_states, received_rewards, 
                                hidden_mind):
        """predicting actions"""
        cuda = self.args.cuda
        cur_states_var  = torch.cat([to_Variable(torch.from_numpy(state).float().unsqueeze(0), cuda) 
                                    for state in cur_states])
        prev_actions_var = torch.cat([to_Variable(expand(one_hot(prev_action, self.env.action_size), 
                                    (self.env.action_size, 
                                     self.env.map_dim[0], self.env.map_dim[1])), cuda)
                                    for prev_action in prev_actions])
        rewards_var = torch.cat([to_Variable(expand(torch.FloatTensor(reward).unsqueeze(0), 
                                    (self.env.nb_goal_types * self.env.nb_pay_types, 
                                     self.env.map_dim[0], self.env.map_dim[1])), cuda)
                                    for reward in received_rewards])

        minds, pred_policies, hidden_mind \
            = model(prev_actions_var, rewards_var, cur_states_var, self.hist_minds, hidden_mind)
        if (self.env.steps + 1) % self.args.t_max == 0: # maximum bptt length
            hidden_mind = self._break_chain(hidden_mind)
        pred_actions = list(self.sampler_argmax.sample(np.exp(pred_policies.cpu().data.numpy())))

        return minds, pred_policies, pred_actions, hidden_mind
    
    
    def act_workers(self, prev_actions, cur_states, received_rewards, 
                       goal_list, payment_list, hidden_mind, verbose=0):
        """act one step in the env"""
        minds, pred_policies, pred_actions, hidden_mind = \
            self.pred_actions_workers(self.mind_tracker, 
                                         prev_actions, 
                                         cur_states, received_rewards, hidden_mind)
        gt_actions = self.select_actions_workers(goal_list, payment_list)
        commits = [agent.goal for agent in self.worker_agents]
        if verbose > 1:
            print('commits:', commits)
        gt_actions_names = [self.env.action_space[gt_action] for gt_action in gt_actions]
        self.env.send_action(list(range(self.env.nb_agents)), gt_actions_names)
        gt_rewards, gt_costs, agents_reached_goal, term = \
            self.env.step(commits=commits, payments=payment_list)    
        # when finished the task, the commitment no longer holds   
        for goal in range(self.env.nb_goal_types):
            for agent_id in agents_reached_goal[goal]:
                self.worker_agents[agent_id].goal = -1

        for agent_id, agent, goal, pay, commit in zip(range(self.env.nb_agents), self.env.agents, goal_list, payment_list, commits):
            self.worker_stats[agent['identity']].update_commit(goal, pay, int(commit == goal))
            if commit != goal: # worker did not sign the contract
                if self.worker_agents[agent_id].last_goal != -1 and self.commit_steps[agent['identity']] > 0: # previous unsuccessful commitment
                    self.worker_stats[agent['identity']].update_success(self.worker_agents[agent_id].last_goal, self.commit_steps[agent['identity']], 0.0)
                self.commit_steps[agent['identity']] = 0
            else: # worker signed the contract
                if goal == self.worker_agents[agent_id].last_goal: # same contract
                    self.commit_steps[agent['identity']] += 1
                else: # new contract
                    self.commit_steps[agent['identity']] = 1
                if agent_id in agents_reached_goal[goal]:
                    self.worker_stats[agent['identity']].update_success(goal, self.commit_steps[agent['identity']], 1.0)
                elif self.env.steps == self.args.max_episode_length: # last time step
                    self.worker_stats[agent['identity']].update_success(goal, self.commit_steps[agent['identity']], 0.0)

        return minds, pred_policies, gt_actions, commits, hidden_mind, gt_rewards, gt_costs, term


    def generate_reward(self, model, cur_states_var, minds):
        """generate rewards based on manager policy"""
        hist_mind_list = list(self.hist_minds.split(1))
        policies_goal, policies_pay, value, value_pay = \
            model(cur_states_var, minds, hist_mind_list)

        goals = self.sampler_egreedy0.sample(policies_goal.data.cpu().numpy(), self.args.cuda)
        payments = self.sampler_egreedy.sample(policies_pay.data.cpu().numpy(), self.args.cuda)
        goal_list = list(goals.data.cpu().numpy().reshape((-1,)))
        payment_list = list(payments.data.cpu().numpy().reshape((-1,)))
        received_rewards = rewards_from_goals_payments(goal_list,
                                                       payment_list,
                                                       self.env.nb_goal_types,
                                                       self.env.nb_pay_types)

        return policies_goal, policies_pay, Variable(goals.data), Variable(payments.data), goal_list, payment_list, \
               received_rewards, value, value_pay


    def act_manager(self, cur_states, minds):
        """act one step by the manager policy"""
        cur_states_var = [to_Variable(torch.from_numpy(state).float().unsqueeze(0), self.args.cuda) 
                                for state in cur_states]
        mind_list = list(minds.split(1))

        policies_goal, policies_pay, goals, payments, goal_list, payment_list, \
        received_rewards, value, value_pay = \
            self.generate_reward(self.manager_module, cur_states_var, mind_list)

        return policies_goal, policies_pay, goals, payments, goal_list, payment_list, \
               received_rewards, value, value_pay


    def rollout(self, nb_agents=2, nb_resources=2, 
                verbose=0, test=False):
        """rollout for an episode"""
        self.env.setup(nb_agents=nb_agents, nb_resources=nb_resources,
                       episode_id=self.episode_id if test else None)
        if self.args.exp_name.startswith('Collection'):
            self.worker_agents = [Collector(agent_id) for agent_id in range(nb_agents)]
        elif self.args.exp_name.startswith('Crafting'):
            self.worker_agents = [Crafter(agent_id) for agent_id in range(nb_agents)]
        else:
            raise ValueError('Invalid environment ({})!'.format(self.args.exp_name))
        if test:
            is_exploring = [False] * self.env.nb_agents
        else:
            is_exploring = [True if random.random() < self.args.eps else False for _ in range(self.env.nb_agents)]
        for agent_id, agent in enumerate(self.env.agents):
            if agent['identity'] not in self.worker_stats:
                self.worker_stats[agent['identity']] = WorkerStats(self.env.nb_goal_types, self.env.nb_pay_types, self.args.window_size,
                                                              self.args.max_episode_length, self.args.lr_stats) 
            self.worker_stats[agent['identity']].add_episode()
            if self.worker_stats[agent['identity']].nb_episodes > self.args.eps_episodes:
                is_exploring[agent_id] = False

        self.commit_steps = {agent['identity']: 0 for agent in self.env.agents} # length of the current commitment of each agent
        # encoding recent history
        self.encode_history_workers()
        
        # rollout trajectory for the on-policy update
        on_policy = {'policies_goal':        [None] * self.args.max_episode_length,
                     'policies_pay':         [None] * self.args.max_episode_length,
                     'goals':                [None] * self.args.max_episode_length,
                     'payments':             [None] * self.args.max_episode_length,
                     'commits':              [None] * self.args.max_episode_length,
                     'value':                [None] * self.args.max_episode_length,
                     'value_pay':            [None] * self.args.max_episode_length,  
                     'pred_policies':        [None] * self.args.max_episode_length,
                     'gt_rewards':           [None] * self.args.max_episode_length,
                     'gt_costs':             [None] * self.args.max_episode_length,
                     'cur_states':           [None] * self.args.max_episode_length, 
                     'gt_actions':           [None] * self.args.max_episode_length, 
                     'received_rewards':     [None] * self.args.max_episode_length, 
                     'done':                 [None] * self.args.max_episode_length, 
                     'mask':                 [None] * self.args.max_episode_length} 

        c_r_all = {goal: 0 for goal in range(self.env.nb_goal_types)} # initial cummulatitve rewards of all goals
        c_c_all = {pay: 0 for pay in range(self.env.nb_pay_types)} # initial commulative costs of levels of payments 

        hidden_mind, hidden_hist = self._create_hidden_states(self.env.nb_agents)

        commits = [-1] * self.env.nb_agents
        cur_states = self.env.get_world_agent_state_all() # initial state
        prev_actions = [self.env.action_size - 1 for _ in range(self.env.nb_agents)]
        
        received_rewards_zero = [[0] * self.env.nb_goal_types * self.env.nb_pay_types for _ in range(self.env.nb_agents)]

        goals = torch.cat([self.sampler_exp.sample(self.worker_stats[agent['identity']].get_prob_commits(), self.args.cuda) 
                                for agent in self.env.agents]) 
        goal_list = list(goals.data.cpu().numpy().reshape((-1,)))
        goal_list = [int(g) for g in goal_list]
        payment_list = [0] * self.env.nb_agents
        received_rewards = rewards_from_goals_payments(goal_list,
                                                       payment_list,
                                                       self.env.nb_goal_types,
                                                       self.env.nb_pay_types)
        minds, pred_policies, gt_actions, commits, hidden_mind, gt_rewards, gt_costs, term = \
                self.act_workers(prev_actions, 
                                    cur_states, received_rewards, goal_list, payment_list, hidden_mind, 
                                    verbose=verbose)

        t = 0
        done = term
        on_policy['policies_goal'][t] = None
        on_policy['policies_pay'][t] = None
        on_policy['goals'][t] = None
        on_policy['payments'][t] = None
        on_policy['commits'][t] = commits
        on_policy['value'][t] = None
        on_policy['value_pay'][t] = None
        on_policy['pred_policies'][t] = pred_policies
        on_policy['gt_rewards'][t] = [gt_rewards[goal] for goal in range(self.env.nb_goal_types)]
        on_policy['gt_costs'][t] = [gt_costs[pay] for pay in range(self.env.nb_pay_types)]
        on_policy['cur_states'][t] = cur_states
        on_policy['gt_actions'][t] = gt_actions
        on_policy['received_rewards'][t] = received_rewards
        on_policy['done'][t] = [done] * self.env.nb_agents
        on_policy['mask'][t] = [1] * self.env.nb_agents

        cur_states = self.env.get_world_agent_state_all()


        while self.env.running:
            if verbose: self.env.print_state()

            if verbose > 1:
                print('is_exploring:', is_exploring)

            policies_goal, policies_pay, new_goals, new_payments, new_goal_list, new_payment_list, \
            new_received_rewards, value, value_pay = \
                self.act_manager(cur_states, minds)
            if t % self.args.interval == 0:
                payments = new_payments
                payment_list = copy.deepcopy(new_payment_list)
                # only update goals when not exploring
                for agent_id in range(self.env.nb_agents):
                    if not is_exploring[agent_id]:
                        goal_list[agent_id] = int(new_goal_list[agent_id])
                        # payment_list[agent_id] = int(new_payment_list[agent_id])
                goals = torch.LongTensor(goal_list).unsqueeze(-1)
                goals = to_Variable(goals.cuda(), self.args.cuda)
                # payments = torch.LongTensor(payment_list).unsqueeze(-1)
                # payments = to_Variable(payments.cuda(), self.args.cuda)

                received_rewards = rewards_from_goals_payments(goal_list,
                                                               payment_list,
                                                               self.env.nb_goal_types,
                                                               self.env.nb_pay_types)
               
            if verbose > 1:
                print('policies_goal:\n', policies_goal.data.cpu().numpy())
                print('goals:\n', goal_list)
                print('policies_pay:\n', policies_pay.data.cpu().numpy())
                print('payments:\n', payment_list)
            minds, pred_policies, gt_actions, commits, hidden_mind, gt_rewards, gt_costs, term = \
                self.act_workers(prev_actions, cur_states, received_rewards, 
                                    goal_list, payment_list, hidden_mind, 
                                    verbose=verbose)

            done = 1 if term or self.env.steps >= self.args.max_episode_length else 0
            
            t = self.env.steps - 1
            on_policy['policies_goal'][t] = policies_goal
            on_policy['policies_pay'][t] = policies_pay
            on_policy['goals'][t] = goals
            on_policy['payments'][t] = payments
            on_policy['commits'][t] = commits
            on_policy['value'][t] = value
            on_policy['value_pay'][t] = value_pay
            on_policy['pred_policies'][t] = pred_policies
            on_policy['gt_rewards'][t] = [gt_rewards[goal] for goal in range(self.env.nb_goal_types)]
            on_policy['gt_costs'][t] = [gt_costs[pay] for pay in range(self.env.nb_pay_types)]
            on_policy['cur_states'][t] = cur_states
            on_policy['gt_actions'][t] = gt_actions
            on_policy['received_rewards'][t] = received_rewards
            on_policy['done'][t] = [done] * self.env.nb_agents
            on_policy['mask'][t] = [1] * self.env.nb_agents

            for goal, reward in gt_rewards.items():
                c_r_all[goal] += reward
            for pay, cnt in gt_costs.items():
                c_c_all[pay] += cnt
            cur_states = self.env.get_world_agent_state_all() # next state
            prev_actions = copy.deepcopy(gt_actions)

            if done:
                break

        # padding
        while t < self.args.max_episode_length - 1:
            t += 1
            on_policy['policies_goal'][t] = policies_goal
            on_policy['policies_pay'][t] = policies_pay
            on_policy['goals'][t] = goals
            on_policy['payments'][t] = payments
            on_policy['commits'][t] = commits
            on_policy['value'][t] = value
            on_policy['value_pay'][t] = value_pay
            on_policy['pred_policies'][t] = pred_policies
            on_policy['gt_rewards'][t] = [gt_rewards[goal] for goal in range(self.env.nb_goal_types)]
            on_policy['gt_costs'][t] = [gt_costs[pay] for pay in range(self.env.nb_pay_types)]
            on_policy['cur_states'][t] = cur_states
            on_policy['gt_actions'][t] = [self.env.action_size - 1] * self.env.nb_agents
            on_policy['received_rewards'][t] = received_rewards_zero
            on_policy['done'][t] = [1] * self.env.nb_agents
            on_policy['mask'][t] = [0] * self.env.nb_agents

        if verbose: self.env.print_state()

        return on_policy, c_r_all, c_c_all


    def train(self, init_checkpoint=0):
        """train the model"""
        args = self.args
        cuda = args.cuda
        nb_episodes = args.max_nb_episodes
        self.hist_module.train()
        self.mind_tracker.train()
        self.manager_module.train()
        self.optimizer_mind    = optim.RMSprop(list(self.mind_tracker.parameters()) + \
                                              list(self.manager_module.parameters()), lr=args.lr_mind)
        self.optimizer_manager = optim.RMSprop(list(self.hist_module.parameters()) + \
                                              list(self.mind_tracker.parameters()) + \
                                              list(self.manager_module.parameters()), lr=args.lr_manager)
        cumulative_rewards = []
        cumulative_costs = []

        if init_checkpoint:
            # load model
            self.load_model(self.hist_module, self.checkpoint_dir + '/checkpoint_hist_' + str(init_checkpoint))
            self.load_model(self.mind_tracker, self.checkpoint_dir + '/checkpoint_mind_' + str(init_checkpoint))
            self.load_model(self.manager_module, self.checkpoint_dir + '/checkpoint_manager_' + str(init_checkpoint))

            # load rewards & costs
            data = pickle.load(open(self.record_dir + '/cumulative_rewards.pik', 'rb'))
            cumulative_rewards = data['cumulative_rewards'][:init_checkpoint]
            cumulative_costs = data['cumulative_costs'][:init_checkpoint]

            # load agent stats
            data = pickle.load(open(self.checkpoint_dir + '/worker_stats_{}.pik'.format(init_checkpoint), 'rb'))
            for identity, agent_stats in data.items():
                self.worker_stats[identity] = WorkerStats(self.env.nb_goal_types, self.env.nb_pay_types, self.args.window_size,
                                                     self.args.max_episode_length, self.args.lr_stats) 
                self.worker_stats[identity].set_stats(agent_stats['nb_commits'], agent_stats['ave_success'], 
                                                   agent_stats['nb_proposals'], agent_stats['ave_commit'], 
                                                   agent_stats['feat'], agent_stats['nb_episodes'])
            # load population
            self.env.load_population(self.checkpoint_dir + '/pop_{}.pik'.format(init_checkpoint))

            start_episode_id = init_checkpoint + 1
        else:
            self.env.generate_population(population_size=args.pop_size)
            start_episode_id = 1

        # update skills, given
        # a list of goals
        # the corresponding of #agents to be updated for each goal
        # and whether to increase or to decrease #agents that have the correspondign skills
        if args.update_skill_pop != 0:
            self.env.update_population(nb_agents_list=args.update_skill_pop,
                                       goal_type_list=args.update_skill_goal,
                                       inc_list=args.update_skill_inc)

        self.sampler_egreedy.reset()

        if args.cuda:
            NLL_loss = nn.NLLLoss(reduce=False).cuda()
            MSE_loss = nn.MSELoss(reduce=False).cuda()
        else:
            NLL_loss = nn.NLLLoss(reduce=False)
            MSE_loss = nn.MSELoss(reduce=False)

        reward_weight_var = to_Variable(torch.FloatTensor(self.env.reward_weight).unsqueeze(0), cuda)
        cost_weight_var = to_Variable(torch.FloatTensor(self.env.cost_weight).unsqueeze(0), cuda)

        for episode_id in range(start_episode_id, nb_episodes):
            self.episode_id = episode_id
            on_policy, c_r_all, c_c_all = \
                self.rollout(nb_resources=args.max_nb_resources, 
                             nb_agents=args.max_nb_agents, verbose=args.verbose)
            cumulative_rewards.append(c_r_all)
            cumulative_costs.append(c_c_all)
            manager_reward = self.env.get_manager_reward(c_r_all)
            manager_cost = self.env.get_manager_cost(c_c_all)
            manager_gain = manager_reward - manager_cost
            print("episode: #{} steps: {} reward: {} manager_gain: {} manager_reward: {} manager_cost: {}".format(episode_id, self.env.steps, c_r_all, manager_gain, manager_reward, manager_cost))

            
            # ===================== on-policy training =====================
            policy_goal_loss, policy_pay_loss = 0, 0
            value_loss = 0
            value_pay_loss = 0
            entropy_goal_loss, entropy_pay_loss = 0, 0
            action_pred_loss = 0
            action_pred_acc = 0
            T = 0
            T_action = 0
            warm_phase = episode_id - start_episode_id + 1 <= args.max_nb_episodes_warm
            for t in range(args.max_episode_length - 1, 0, -1):
                policies_goal = on_policy['policies_goal'][t]
                policies_pay  = on_policy['policies_pay'][t]
                goals     = on_policy['goals'][t]
                payments  = on_policy['payments'][t]
                value     = on_policy['value'][t]
                value_pay = on_policy['value_pay'][t]
                gt_reward = on_policy['gt_rewards'][t]
                gt_cost   = on_policy['gt_costs'][t]
                done      = on_policy['done'][t][0]
                mask      = on_policy['mask'][t][0]
                if mask == 0: continue

                gt_reward = torch.FloatTensor(gt_reward).unsqueeze(0)
                gt_cost = torch.FloatTensor(gt_cost).unsqueeze(0)
                gt_debt = torch.FloatTensor(gt_cost).unsqueeze(0) #for warm up training phase only

                if done: 
                    Vret = gt_reward
                    Vret_pay = gt_cost -(gt_debt if warm_phase else 0)
                else:
                    Vret = gt_reward + Vret * args.discount
                    Vret_pay = gt_cost - (gt_debt if warm_phase else 0) + Vret_pay * args.discount
                if (t - 1) % args.interval == 0:
                    A = torch.dot(to_Variable(Vret, cuda) - value, reward_weight_var) \
                            - torch.dot(to_Variable(Vret_pay, cuda) - value_pay, cost_weight_var)
                    log_prob_goal = (policies_goal.gather(1, goals) + args.min_prob).log()
                    log_prob_payment = (policies_pay.gather(1, payments) + args.min_prob).log()
                    single_step_policy_goal_loss = -(log_prob_goal * float(A.data.cpu().numpy())).mean(0)
                    policy_goal_loss += single_step_policy_goal_loss
                    single_step_policy_pay_loss = -(log_prob_payment * float(A.data.cpu().numpy())).mean(0)
                    policy_pay_loss += single_step_policy_pay_loss
                    value_loss += MSE_loss(value, to_Variable(Vret, cuda)).mean(-1).mean(0) / 2
                    value_pay_loss += MSE_loss(value_pay, to_Variable(Vret_pay, cuda)).mean(-1).mean(0) / 2
                    entropy_goal_loss += ((policies_goal + args.min_prob).log() * policies_goal).sum(1).mean(0)
                    entropy_pay_loss += ((policies_pay + args.min_prob).log() * policies_pay).sum(1).mean(0)
                    T += 1
                gt_actions_var = to_Variable(torch.LongTensor(on_policy['gt_actions'][t]), cuda)
                action_pred_loss += NLL_loss(on_policy['pred_policies'][t], gt_actions_var)
                pred_actions = list(self.sampler_argmax.sample(np.exp(on_policy['pred_policies'][t].cpu().data.numpy())))
                action_pred_acc += (pred_actions == gt_actions_var.data.cpu().numpy()).sum()
                T_action += len(on_policy['mask'][t])
            
            policy_goal_loss /= T
            policy_pay_loss /= T
            value_loss /= T
            value_pay_loss /= T
            entropy_goal_loss /= T
            entropy_pay_loss /= T
            action_pred_loss = action_pred_loss.sum() / T_action
            action_pred_acc = action_pred_acc / T_action
            action_pred_loss_value = action_pred_loss.data.cpu().numpy()[0]
            if args.verbose > 1:
                print('policy_goal_loss:', policy_goal_loss.data.cpu().numpy()[0])
                print('policy_pay_loss:', policy_pay_loss.data.cpu().numpy()[0])
                print('value_loss:', value_loss.data.cpu().numpy()[0])
                print('value_pay_loss:', value_pay_loss.data.cpu().numpy()[0])
                print('entropy_goal_loss:', entropy_goal_loss.data.cpu().numpy()[0])
                print('entropy_pay_loss:', entropy_pay_loss.data.cpu().numpy()[0])            
                print('action_pred_loss:', action_pred_loss_value)
                print('action_pred_acc:', action_pred_acc)

            update_network(self.args, 
                           self.hist_module, self.mind_tracker, self.manager_module, 
                           policy_goal_loss + policy_pay_loss + \
                           value_loss + value_pay_loss + \
                           (entropy_goal_loss + entropy_pay_loss) * args.entropy_weight + \
                           action_pred_loss * args.IL, 
                           self.optimizer_manager)

            self.sampler_egreedy.update()

            if (episode_id + 1) % args.checkpoint_episodes == 0:
                # save network parameters
                self.save_model(self.hist_module,
                                self.checkpoint_dir + "/checkpoint_hist_" + str(episode_id + 1))
                self.save_model(self.mind_tracker, 
                                self.checkpoint_dir + "/checkpoint_mind_" + str(episode_id + 1))   
                self.save_model(self.manager_module,
                                self.checkpoint_dir + "/checkpoint_manager_" + str(episode_id + 1))
                # save woker agents' stats
                pickle.dump({identity: {'nb_commits': agent_stats.nb_commits,
                                        'ave_success': agent_stats.ave_success,
                                        'nb_proposals': agent_stats.nb_proposals,
                                        'ave_commit': agent_stats.ave_commit,
                                        'feat': agent_stats.feat,
                                        'nb_episodes': nb_episodes} \
                                    for identity, agent_stats in self.worker_stats.items()},
                        open(self.checkpoint_dir + '/worker_stats_{}.pik'.format(episode_id + 1), 'wb'), 
                        protocol=pickle.HIGHEST_PROTOCOL)
                # save population info
                self.env.save_population(self.checkpoint_dir + '/pop_{}.pik'.format(episode_id + 1))
            if (episode_id + 1) % 1000 == 0:
                pickle.dump({'cumulative_rewards':cumulative_rewards, 'cumulative_costs':cumulative_costs},
                            open(self.record_dir + '/cumulative_rewards.pik', 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
                

    def test(self, checkpoint=0):
        """test the model"""
        nb_episodes = 100
        self.hist_module.eval()
        self.mind_tracker.eval()
        self.manager_module.eval()
        args = self.args
        if checkpoint:
            self.load_model(self.hist_module, self.checkpoint_dir + '/checkpoint_hist_' + str(checkpoint))
            self.load_model(self.mind_tracker, self.checkpoint_dir + '/checkpoint_mind_' + str(checkpoint))
            self.load_model(self.manager_module, self.checkpoint_dir + '/checkpoint_manager_' + str(checkpoint))

            # load agent stats
            data = pickle.load(open(self.checkpoint_dir + '/worker_stats_{}.pik'.format(checkpoint), 'rb'))
            for identity, agent_stats in data.items():
                self.worker_stats[identity] = WorkerStats(self.env.nb_goal_types, self.env.nb_pay_types, self.args.window_size,
                                                       self.args.max_episode_length, self.args.lr_stats) 
                self.worker_stats[identity].set_stats(agent_stats['nb_commits'], agent_stats['ave_success'], 
                                                   agent_stats['nb_proposals'], agent_stats['ave_commit'], 
                                                   agent_stats['feat'], agent_stats['nb_episodes'])
            # load population if reusing old populuation
            self.env.load_population(self.checkpoint_dir + '/pop_{}.pik'.format(checkpoint))
        else:
            raise ValueError('Please specify a valid checkpoint!')
        self.sampler_egreedy.set_zero()
        self.sampler_exp.set_zero()           

        manager_gain_list = [0] * nb_episodes
        steps_list = [0] * nb_episodes

        for episode_id in range(nb_episodes):
            self.episode_id = episode_id
            on_policy, c_r_all, c_c_all = self.rollout(nb_resources=args.max_nb_resources, 
                                                       nb_agents=args.test_max_nb_agents, 
                                                       verbose=args.verbose,
                                                       test=True)
            manager_reward = self.env.get_manager_reward(c_r_all)
            manager_cost = self.env.get_manager_cost(c_c_all)
            manager_gain = manager_reward - manager_cost
            manager_gain_list[episode_id] = manager_gain
            print("episode: #{} steps: {} reward: {} manager_gain: {} manager_reward: {} manager_cost: {}".format(episode_id, self.env.steps, c_r_all, manager_gain, manager_reward, manager_cost))
            steps_list[episode_id] = self.env.steps

        ave_gain = sum(manager_gain_list) / nb_episodes
        print('average reward:', ave_gain)
        ave_steps = sum(steps_list) / nb_episodes
        print('average steps:', ave_steps)
        return ave_gain


    def test_new_population(self, checkpoint=0):
        """test the model on a new and changing population"""
        nb_episodes = 10000
        self.hist_module.eval()
        self.mind_tracker.eval()
        self.manager_module.eval()
        args = self.args
        if checkpoint:
            self.load_model(self.hist_module, self.checkpoint_dir + '/checkpoint_hist_' + str(checkpoint))
            self.load_model(self.mind_tracker, self.checkpoint_dir + '/checkpoint_mind_' + str(checkpoint))
            self.load_model(self.manager_module, self.checkpoint_dir + '/checkpoint_manager_' + str(checkpoint))
        else:
            raise ValueError('Please specify a valid checkpoint!')
        self.sampler_egreedy.set_zero()
        self.sampler_exp.set_zero()

        cumulative_rewards, cumulative_costs = [], []
        manager_gain_list = [0] * nb_episodes
        steps_list = [0] * nb_episodes

        self.env.generate_population(population_size=args.test_pop_size)
        nb_new_agents = int(args.test_pop_size * args.test_update_pct)

        for episode_id in range(nb_episodes):
            self.episode_id = episode_id
            on_policy, c_r_all, c_c_all = self.rollout(nb_resources=args.max_nb_resources, 
                                                       nb_agents=args.test_max_nb_agents, 
                                                       verbose=args.verbose,
                                                       test=True)
            cumulative_rewards.append(c_r_all)
            cumulative_costs.append(c_c_all)
            manager_reward = self.env.get_manager_reward(c_r_all)
            manager_cost = self.env.get_manager_cost(c_c_all)
            manager_gain = manager_reward - manager_cost
            manager_gain_list[episode_id] = manager_gain
            print("episode: #{} steps: {} reward: {} manager_gain: {} manager_reward: {} manager_cost: {}".\
                  format(episode_id, self.env.steps, c_r_all, manager_gain, manager_reward, manager_cost))
            steps_list[episode_id] = self.env.steps
            # replacing old agents with new ones periodically
            if (episode_id + 1) % args.test_update_freq == 0:
                identities = random.sample(range(args.test_pop_size), nb_new_agents)
                for identity in identities:
                    self.worker_stats[identity].reset_stats()
        pickle.dump({'cumulative_rewards':cumulative_rewards, 'cumulative_costs':cumulative_costs},
                    open(self.record_dir + '/test_popsize{}_eps{}_freq{}_change{}_cumulative_rewards.pik'.\
                         format(args.test_pop_size, args.eps_episodes, args.test_update_freq, args.test_update_pct), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

        ave_gain = sum(manager_gain_list) / nb_episodes
        print('average reward:', ave_gain)
        ave_steps = sum(steps_list) / nb_episodes
        print('average steps:', ave_steps)


    def _create_hidden_states(self, n):
        if self.args.lstm_mind:
            cuda = self.args.cuda
            hx_mind = to_Variable(torch.zeros(n, self.args.mind_dim), cuda)
            cx_mind = to_Variable(torch.zeros(n, self.args.mind_dim), cuda)
            hx_hist = to_Variable(torch.zeros(n, self.args.hist_dim), cuda)
            cx_hist = to_Variable(torch.zeros(n, self.args.hist_dim), cuda)
        hidden_mind = (hx_mind, cx_mind) if self.args.lstm_mind else None
        hidden_hist = (hx_hist, cx_hist) if self.args.lstm_hist else None
        return hidden_mind, hidden_hist


    def _break_chain(self, hidden_state):
        """break the BPTT chain"""
        if hidden_state:
            return (Variable(hidden_state[0].data), Variable(hidden_state[1].data))
        else:
            return None


    def save_model(self, model, path):
        """save trained model parameters"""
        torch.save(model.state_dict(), path)


    def load_model(self, model, path, avoid = None):
        """load trained model parameters"""
        state_dict = dict(torch.load(path))
        if avoid is not None:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(avoid)}
            model_dict = model.state_dict()
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(state_dict)
