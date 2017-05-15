# -*- coding: utf-8 -*-
# file: model_utils.py
# author: JinTian
# time: 10/05/2017 6:07 PM
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
import os
import glob
import numpy as np
import time
import math


def load_previous_model(model, checkpoints_dir, model_prefix):
    f_list = glob.glob(os.path.join(checkpoints_dir, model_prefix) + '-*.pth')
    start_epoch = 1
    if len(f_list) >= 1:
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        last_checkpoint = f_list[np.argmax(epoch_list)]
        start_epoch = np.max(epoch_list)
        if os.path.exists(last_checkpoint):
            print('load from {}'.format(last_checkpoint))
            model.load_state_dict(torch.load(last_checkpoint, map_location=lambda storage, loc: storage))
    return model, start_epoch


def save_model(model, checkpoints_dir, model_prefix, epoch, max_keep=5):

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    f_list = glob.glob(os.path.join(checkpoints_dir, model_prefix) + '-*.pth')
    if len(f_list) >= max_keep + 2:
        # this step using for delete the more than 5 and litter one
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
        for f in to_delete:
            os.remove(f)
    name = model_prefix + '-{}.pth'.format(epoch)
    file_path = os.path.join(checkpoints_dir, name)
    torch.save(model.state_dict(), file_path)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return 'cost: %s,  estimate: %s %s ' % (as_minutes(s), as_minutes(rs), str(round(percent*100, 2)) + '%')
