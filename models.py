# -*- coding: utf-8 -*-
# file: models.py
# author: JinTian
# time: 14/05/2017 3:20 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.global_config import *


class ConditionalRNN(nn.Module):
    """
    here we construct a Conditional RNN, circle neural networks
    This model's graph can see from here:
    https://i.imgur.com/jzVrf7f.png
    """

    def __init__(self, conditional_input_size, input_size, hidden_size, output_size):
        super(ConditionalRNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(conditional_input_size + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(conditional_input_size + input_size + hidden_size, output_size)

        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        # self.dropout = nn.Dropout(0.1)
        self.soft_max = nn.LogSoftmax()

    def forward(self, conditional_inputs, inputs, hidden):
        inputs_combined = torch.cat((conditional_inputs, inputs, hidden), 1)
        hidden = self.i2h(inputs_combined)
        if USE_GPU:
            hidden = hidden.cuda()
        outputs = self.i2o(inputs_combined)

        outputs_combined = torch.cat((hidden, outputs), 1)
        outputs = self.o2o(outputs_combined)
        # outputs = self.dropout(outputs)
        outputs = self.soft_max(outputs)
        if USE_GPU:
            outputs = outputs.cuda()
        return outputs, hidden

    def init_hidden(self):
        """
        initial hidden for the very first inputs along with the real data inputs
        :return:
        """
        hidden = Variable(torch.zeros(1, self.hidden_size))
        if USE_GPU:
            hidden = hidden.cuda()
        return hidden
