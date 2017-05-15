# -*- coding: utf-8 -*-
# file: data_loader.py
# author: JinTian
# time: 14/05/2017 11:59 AM
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
import os
import glob
import random
import torch
from torch.autograd import Variable
from utils.clean_cn import clean_cn_corpus
from utils.global_config import *
import numpy as np


class DataLoader(object):

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self._init_vital_hold()

    def _init_vital_hold(self):
        self.category_lines = {}
        for file_name in glob.glob('datasets/names/*.name'):
            data_set = file_name.split('/')[-1].split('.')[0]
            self.category_lines[data_set] = clean_cn_corpus(file_name)
        self.vocab_pool = []
        for l in self.category_lines.values():
            for item in l:
                for i in item:
                    if i not in self.vocab_pool:
                        self.vocab_pool.append(i)
        # vocab_pool = list(set(vocab_pool))
        self.vocab_pool.append('<EOS>')
        print('vocab pool size: ', len(self.vocab_pool))

    def load_data(self):
        random_category = np.random.choice(list(self.category_lines.keys()))
        random_line = np.random.choice(list(self.category_lines[random_category]))
        print('random pair: category: {}, line: {}'.format(random_category, random_line))

        category_tensor = Variable(self.category_to_tensor(list(self.category_lines.keys()), random_category))
        inputs_tensor = Variable(self.inputs_to_tensor(self.vocab_pool, random_line))
        target_tensor = Variable(self.target_to_tensor(self.vocab_pool, random_line))

        # print('category data: ', category_tensor.cpu().data.numpy())
        # print('inputs data: ', inputs_tensor.cpu().data.numpy())
        # print('target data: ', target_tensor.cpu().data.numpy())
        return category_tensor, inputs_tensor, target_tensor

    @staticmethod
    def category_to_tensor(all_category, category):
        """
        one hot to tensor, return something like [[0, 1]]
        :param all_category:
        :param category:
        :return:
        """
        assert isinstance(all_category, list), 'all category must be a list'
        tensor = torch.zeros(1, len(all_category))
        li = all_category.index(category)
        tensor[0][li] = 1
        if USE_GPU:
            tensor = tensor.cuda()
        return tensor

    @staticmethod
    def inputs_to_tensor(vocab_pool, line):
        assert isinstance(vocab_pool, list), 'vocab_pool must be a list'
        tensor = torch.zeros(len(line), 1, len(vocab_pool))
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][vocab_pool.index(letter)] = 1
        if USE_GPU:
            tensor = tensor.cuda()
        return tensor

    @staticmethod
    def target_to_tensor(vocab_pool, line):
        assert isinstance(vocab_pool, list), 'vocab_pool must be a list'
        letter_indexes = [vocab_pool.index(line[li]) for li in range(1, len(line))]
        letter_indexes.append(len(vocab_pool) - 1)  # EOS is the last item in vocab pool
        tensor = torch.LongTensor(letter_indexes)
        if USE_GPU:
            tensor = tensor.cuda()
        return tensor

    @staticmethod
    def is_chinese(uchar):
        """is chinese"""
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        else:
            return False





