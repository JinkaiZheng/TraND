#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : logger.py
@Author: Jinkai Zheng
@Date  : 2020/1/11 14:51
@E-mail  : zhengjinkai3@qq.com
'''

import logging
import os
from configs import parser_argument

cfg = parser_argument()

class Logger:
    def __init__(self, name=__name__):
        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.DEBUG)

        log_path = os.path.join('outputs/TraND', cfg.log_name)
        if not os.path.exists(log_path): os.makedirs(log_path)
        logname = log_path + '/' + cfg.log_name + '.log'
        fh = logging.FileHandler(logname, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]   %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        return self.logger


log = Logger(__name__).get_log
