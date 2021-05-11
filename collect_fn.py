#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : sth_from_GaitSet.py
@Author: Jinkai Zheng
@Date  : 2019/12/30 14:40
@E-mail  : zhengjinkai3@qq.com
'''

import math
import random
import numpy as np
from configs import parser_argument

cfg = parser_argument()


def train_collate_fn(batch):

    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]
    index = [batch[i][5] for i in range(batch_size)]
    batch = [seqs, view, seq_type, label, None, index]

    def select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        frame_id_list = random.choices(frame_set, k=30)
        _ = [feature.loc[frame_id_list].values for feature in sample]
        return _

    seqs = list(map(select_frame, range(len(seqs))))
    seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    batch[0] = seqs

    return batch


def test_collate_fn(batch):
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]
    index = [batch[i][5] for i in range(batch_size)]
    batch = [seqs, view, seq_type, label, None, index]

    def select_frame(index):
        sample = seqs[index]
        _ = [feature.values for feature in sample]
        return _

    seqs = list(map(select_frame, range(len(seqs))))

    gpu_num = min(len(cfg.gpus.split(',')), batch_size)
    batch_per_gpu = math.ceil(batch_size / gpu_num)
    batch_frames = [[
                        len(frame_sets[i])
                        for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                        if i < batch_size
                    ] for _ in range(gpu_num)]
    if len(batch_frames[-1]) != batch_per_gpu:
        for _ in range(batch_per_gpu - len(batch_frames[-1])):
            batch_frames[-1].append(0)
    max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
    seqs = [[
                np.concatenate([
                                    seqs[i][j]
                                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                    if i < batch_size
                                    ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
    seqs = [np.asarray([
                            np.pad(seqs[j][_],
                                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                   'constant',
                                   constant_values=0)
                            for _ in range(gpu_num)])
            for j in range(feature_num)]
    batch[4] = np.asarray(batch_frames)
    batch[0] = seqs

    return batch
