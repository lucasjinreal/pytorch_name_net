# -*- coding: utf-8 -*-
# file: train.py
# author: JinTian
# time: 10/05/2017 8:24 PM
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
"""
we are going to train a RNN Network to generate names.
etc. you given 公司名字
then network will generate name such as:
微软 谷歌 苹果...
"""
from data_loader import DataLoader
import os
import sys
import numpy as np
from utils.model_utils import *
from models import ConditionalRNN
from torch import nn
from torch.autograd import Variable
from utils.global_config import *


def train_model(model, category_tensor, inputs_tensor, target_tensor, criterion, lr):
    hidden = model.init_hidden()
    model.zero_grad()
    loss = 0

    for i in range(inputs_tensor.size()[0]):
        output, hidden = model.forward(category_tensor, inputs_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()

    for p in model.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.data[0] / inputs_tensor.size()[0]


def train():
    data_loader = DataLoader()
    model = ConditionalRNN(
        conditional_input_size=len(data_loader.category_lines.keys()),
        input_size=len(data_loader.vocab_pool),
        hidden_size=64,
        output_size=len(data_loader.vocab_pool)
    )
    if USE_GPU:
        model = model.cuda()

    start = time.time()
    print('start training...')
    criterion = nn.NLLLoss()

    model, start_epoch = load_previous_model(model, checkpoints_dir=checkpoints_dir, model_prefix=model_prefix)
    print(start_epoch)
    try:
        for epoch in range(start_epoch, n_epochs + 1):
            train_bundle = data_loader.load_data()
            output, loss = train_model(model, *train_bundle, criterion=criterion, lr=lr)
            if epoch % print_every == 0:
                print('Epoch: %s, %s loss: %.4f' % (epoch, time_since(start, epoch/n_epochs), loss))
            if epoch % save_every == 0:
                save_model(model, checkpoints_dir=checkpoints_dir, model_prefix=model_prefix, epoch=epoch)
            if epoch % sample_every == 0:
                print(random_sample_evaluate(data_loader, model, 'human', start_word='王'))
                print(random_sample_evaluate(data_loader, model, 'company', start_word='欧'))
                print(random_sample_evaluate(data_loader, model, 'human', start_word='吴'))
                print(random_sample_evaluate(data_loader, model, 'human', start_word='金'))
                print(random_sample_evaluate(data_loader, model, 'company', start_word='万'))
                print(random_sample_evaluate(data_loader, model, 'human', start_word='张'))
    except KeyboardInterrupt:
        print('interrupted...')
        save_model(model, checkpoints_dir=checkpoints_dir, model_prefix=model_prefix, epoch=epoch, max_keep=10)


def random_sample_evaluate(data_loader, model, category, start_word='金'):
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


def main():
    train()


if __name__ == '__main__':
    main()