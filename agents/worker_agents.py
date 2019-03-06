# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random

class Collector():
    """Collector agents"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.goal = -1
        self.last_goal = -1


    def select_action(self, env, assigned_goal, assigned_pay, noise=0):
        max_reward = 0
        self.last_goal = self.goal
        self.goal = -1
        selected_goal = []
        selected_action = dict()
        for goal in range(env.nb_goal_types):
            cost = 1 if 'cost' not in env.full_population[env.agents[self.agent_id]['identity']] \
                     else env.full_population[env.agents[self.agent_id]['identity']]['cost']
            reward = env.resource_weights[self.agent_id][goal] * cost + (assigned_pay + 1 if assigned_goal == goal else 0)
            if reward < 1e-6:
                continue
            steps, actions = env.search_path(30, goal, actionable_agents=[self.agent_id])
            selected_action[goal] = actions[0][0]
            if steps > -1:
                if reward > max_reward:
                    max_reward = reward
                    selected_goal = [goal]
                elif reward == max_reward and reward > 0:
                    selected_goal.append(goal)
        if selected_goal:
            if assigned_goal in selected_goal and len(selected_goal) == 1:
                self.goal = assigned_goal
                final_goal = assigned_goal
            else:
                if assigned_goal in selected_goal:
                    selected_goal.remove(assigned_goal)
                final_goal = random.choice(selected_goal)
            return random.choice([0, 1, 2, 4]) if random.random() < noise else selected_action[final_goal]
        return random.choice([0, 1, 2, 4]) if random.random() < noise else env.action_size - 1


class Crafter():
    """Crafter agents"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.goal = -1
        self.last_goal = -1


    def select_action(self, env, assigned_goal, assigned_pay, noise=0):
        max_reward = 0
        self.last_goal = self.goal
        self.goal = -1
        selected_goal = []
        selected_action = dict()
        assigned_pay *= env.cost_weight[assigned_pay]

        cost = 1 if 'cost' not in env.full_population[env.agents[self.agent_id]['identity']] \
                     else env.full_population[env.agents[self.agent_id]['identity']]['cost']
        # first check the desired goal
        goal = env.agents[self.agent_id]['desire']

        steps, actions = env.search_path(30, goal, actionable_agents=[self.agent_id])
        if steps == -1:
            action = env.action_size - 1
        else:
            action = actions[0][0]
        if assigned_pay == 0 or steps > -1 and assigned_pay <= cost: # not enough payment
            return random.choice([0, 1, 2, 5]) if random.random() < noise else action

        if assigned_goal != goal:
            steps, actions = env.search_path(30, assigned_goal, actionable_agents=[self.agent_id])
        if steps == -1:
            action = env.action_size - 1
        else:
            action = actions[0][0]
            self.goal = assigned_goal

        return random.choice([0, 1, 2, 5]) if random.random() < noise else action

