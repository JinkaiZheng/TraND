#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : all_configs.py
@Author: Jinkai Zheng
@Date  : 2019/12/30 20:09
@E-mail  : zhengjinkai3@qq.com
'''

import argparse
import time

def parser_argument():
    parser = argparse.ArgumentParser()
    # timestamp
    # stt = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    # tt = int(time.time())
    # dataset to be used
    parser.add_argument('--cfgs', type=str, nargs='*',
                            help='config files to load')

    # lr policy to be used
    parser.add_argument('--lr-policy', default='step', type=str,
                     help='lr policy to be used. (default: step)')

    # args for protocol
    # parser.add_argument('--protocol', default='knn', type=str,
    #                  help='protocol used to validate model')

    # args for network training
    parser.add_argument('--max-epoch', default=300, type=int,
                     help='max epoch per round. (default: 200)')
    parser.add_argument('--max-round', default=5, type=int,
                     help='max iteration, including initialisation one. '
                          '(default: 5)')
    parser.add_argument('--iter-size', default=1, type=int,
                     help='caffe style iter size. (default: 1)')
    parser.add_argument('--display-freq', default=20, type=int,
                     help='display step')
    parser.add_argument('--gpus', default='', type=str,
                        help='available gpu list. (default: \'\')')
    parser.add_argument('--source', default='casia-b', type=str,
                        help='source dataset to be used. (default: casia-b)')
    parser.add_argument('--target', default='oulp', type=str,
                        help='target dataset to be used. (default: oulp)')
    parser.add_argument('--ANs-select-rate', default=0.25, type=float)
    parser.add_argument('--loss-init-rate', default=0.1, type=float)
    parser.add_argument('--ANs-size', default=1, type=int)
    parser.add_argument('--log-name', default='logloglog', type=str)
    parser.add_argument('--resize', default=32)
    parser.add_argument('--start-round', default=0, type=int)
    parser.add_argument('--base-lr', default=0.00001)
    parser.add_argument('--lr-decay-offset', default=80)
    parser.add_argument('--lr-decay-step', default=40)
    parser.add_argument('--lr-decay-rate', default=0.1)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--low-dim', default=62*256,
                        help='GaitSet model\'s output is (62,256)D, '
                             'then concated by 2 seq and gei features, last they need to be viewed')
    parser.add_argument('--npc-temperature', default=0.1, type=float)
    parser.add_argument('--npc-momentum', default=0.5, type=float)
    parser.add_argument('--load-pretrained', default=False,
                        help='decide whether load source domain\'s model parameters')
    cfg = parser.parse_args()

    return cfg