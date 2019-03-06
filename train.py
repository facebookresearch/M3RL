# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import torch

from agents.M3RL import M3RL

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--t-max', type=int, default=30, help='Max number of forward steps for A2C before update')
parser.add_argument('--max-episode-length', type=int, default=30, help='Maximum episode length')
parser.add_argument('--discount', type=float, default=0.95, help='Discount factor')
parser.add_argument('--max-gradient-norm', type=float, default=1, help='Max value of gradient L1 norm for gradient clipping')
parser.add_argument('--lr-mind', type=float, default=0.001, help='Learning rate for hist and mind tracker modules')
parser.add_argument('--lr-manager', type=float, default=0.0003, help='Learning rate for the manager module')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory of checkpoints')
parser.add_argument('--record-dir', type=str, default='record', help='Directory record')
parser.add_argument('--init-epsilon', type=float, default=0.1, help='Initial e-greedy coefficient')
parser.add_argument('--final-epsilon', type=float, default=0.0, help='Final e-greedy coefficient')
parser.add_argument('--checkpoint-episodes', type=int, default=10000, help='Frequency of saving checkpoints')
parser.add_argument('--max-exp-steps', type=int, default=100000, help='Maximum steps for exploration')
parser.add_argument('--nb-channels', type=int, default=64, help='Number of channels in state encoder')
parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension in state encoder')
parser.add_argument('--exp-name', type=str, default='Collection_v0', help='Experiment name')
parser.add_argument('--min-prob', type=float, default=1e-6, help='Minimum policy prob')
parser.add_argument('--init-checkpoint', type=int, default=0, help='Pretrained checkpoint')
parser.add_argument('--fusion', type=str, default='att', help='Type of fusion for goal')
parser.add_argument('--att-norm', action='store_true', default=False, help='Whether normalize attention by sigmoid')
parser.add_argument('--pooling', type=str, default='average', help='Type of pooling: average, sum')
parser.add_argument('--mind-dim', type=int, default=128, help='Dimension of the mental state')
parser.add_argument('--hist-dim', type=int, default=128, help='Dimension of the history encoder')
parser.add_argument('--lstm-mind', type=int, default=1, help='1 - LSTM, 0 - MLP')
parser.add_argument('--lstm-hist', type=int, default=1, help='1 - LSTM, 0 - MLP')
parser.add_argument('--verbose', type=int, default=0, help='0 - no game status, 1 - game status, 2 - detailed info')
parser.add_argument('--nb-pay-types', type=int, default=3, help='Number of pay types')
parser.add_argument('--nb-agent-types', type=int, default=4, help='Maximum number of agent types')
parser.add_argument('--nb-resource-types', type=int, default=2, help='Number of resource types')
parser.add_argument('--policy-activation', type=str, default='softmax', help='sigmoid, linear, softmax')
parser.add_argument('--interval', type=int, default=1, help='Commitment constraint')
parser.add_argument('--max-nb-resources', type=int, default=4, help='Maximum number of resources')
parser.add_argument('--max-nb-agents', type=int, default=4, help='Maximum number of agents')
parser.add_argument('--max-nb-episodes', type=int, default=250000, help='Maximum number of training episodes')
parser.add_argument('--include-type', action='store_true', default=False, help='Whether to include identity info in states')
parser.add_argument('--entropy-weight', type=float, default=0.01, help='Weight for the entropy regularization')
parser.add_argument('--pop-size', type=int, default=100, help='Size of the whole simulated populuation')
parser.add_argument('--lr-stats', type=float, default=0.1, help='Stats update rate')
parser.add_argument('--window-size', type=int, default=1, help='Window size for estimating return in game stats')
parser.add_argument('--eps', type=float, default=0.1, help='Agent-wise exploration coefficient')
parser.add_argument('--update-skill-goal', type=int, nargs = '*', default=0, help='Specify a goal to be added as a new skill')
parser.add_argument('--update-skill-pop', type=int, nargs = '*', default=0, help='How many agents to be updated')
parser.add_argument('--update-skill-inc', type=int, nargs = '*', default=0, help='To add (1) or remove (0) skills')
parser.add_argument('--eps-episodes', type=int, default=10000, help='Maximum exploration episodes (for agent-wise e-greedy)')
parser.add_argument('--max-nb-episodes-warm', type=int, default=0, help='Maximum #episodes of warming up training phase')
parser.add_argument('--obstacle', action='store_true', default=False, help='False - no obstacles; True - with obstacles')
parser.add_argument('--noise', type=float, default=0.0, help='How many noisy actions applied to the worker policies')
parser.add_argument('--IL', action='store_true', default=False, help='False - no imitation learning; True - with IL')


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print ('Options')
    print ('=' * 30)
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print ('=' * 30)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    agent = M3RL(args)
    agent.train(args.init_checkpoint)
