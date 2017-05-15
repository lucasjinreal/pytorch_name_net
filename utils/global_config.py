# -*- coding: utf-8 -*-
# file: global_config.py
# author: JinTian
# time: 14/05/2017 6:14 PM
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


USE_GPU = torch.cuda.is_available()

n_epochs = 50000
print_every = 100
save_every = 5000
sample_every = 200
lr = 0.001
checkpoints_dir = './checkpoints'
model_prefix = 'name_net'
