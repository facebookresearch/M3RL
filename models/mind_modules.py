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


class MindTracker(torch.nn.Module):
    """
    mind tracker
    state_action_embed -> mind
    """
    def __init__(self,
                 state_action_embed_dim = 128,
                 mind_dim = 128,
                 lstm = True):
        super(MindTracker, self).__init__()

        self.state_action_embed_dim = state_action_embed_dim
        self.mind_dim = mind_dim
        self.lstm = lstm

        if lstm:
            self.mind_lstm = nn.LSTMCell(state_action_embed_dim, mind_dim)
        else:
            self.mind_fc = nn.Linear(state_action_embed_dim, mind_dim)

        self.apply(weights_init)
        if lstm:
            self.mind_lstm.bias_ih.data.fill_(0)
            self.mind_lstm.bias_hh.data.fill_(0)

    
    def forward(self, 
                state_action_embed, 
                hidden = None):
        if self.lstm:
            (hx, cx) = self.mind_lstm(state_action_embed, hidden)
            return hx, (hx, cx)
        else:
            mind = self.mind_fc(state_action_embed)
            return mind, None
