# -*- coding: utf-8 -*-
# file: generate.py
# author: JinTian
# time: 14/05/2017 8:33 PM
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
from data_loader import DataLoader
import os
import sys
import numpy as np
from utils.model_utils import *
from models import ConditionalRNN
from torch import nn
from torch.autograd import Variable
from utils.global_config import *
import sys


def predict(data_loader, model, category, start_word='金'):
    category_tensor = Variable(data_loader.category_to_tensor(list(data_loader.category_lines.keys()), category))
    inputs = Variable(data_loader.inputs_to_tensor(data_loader.vocab_pool, start_word))
    hidden = model.init_hidden()

    output_name = start_word
    for i in range(20):
        output, hidden = model.forward(category_tensor, inputs[0], hidden)
        top_v, top_i = output.data.topk(1)
        top_i = top_i[0][0]
        if top_i == len(data_loader.vocab_pool) - 1:
            break
        else:
            letter = data_loader.vocab_pool[top_i]
            output_name += letter
        inputs = Variable(data_loader.inputs_to_tensor(data_loader.vocab_pool, letter))
    return output_name


def generate():
    data_loader = DataLoader()

    model = ConditionalRNN(
        conditional_input_size=len(data_loader.category_lines.keys()),
        input_size=len(data_loader.vocab_pool),
        hidden_size=64,
        output_size=len(data_loader.vocab_pool)
    )
    if USE_GPU:
        model = model.cuda()
    model, _ = load_previous_model(model, checkpoints_dir=checkpoints_dir, model_prefix=model_prefix)

    if len(sys.argv) < 2:
        print('usage: python3 generate.py human 赵 to generate a name.')
    else:
        category = sys.argv[1]
        start_word = sys.argv[2]
        output_name = predict(data_loader, model, category, start_word)
        print('generated name for a {}: {}'.format(category, output_name))


if __name__ == '__main__':
    generate()



