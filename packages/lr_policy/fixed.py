#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-26 20:15:47
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

from logger import log
from configs import parser_argument

cfg = parser_argument()

class FixedPolicy:

    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            self.__init_by_cfg()
        else:
            self.__init(*args, **kwargs)

    def __init(self, base_lr):
        self.base_lr = base_lr
        log.debug('Going to use [fixed] learning policy for optimization'
            ' with base learning rate [%.5f]' % base_lr)

    def __init_by_cfg(self):
        self.__init(cfg.base_lr)

    def update(self, epoch, *args, **kwargs):
        return self.base_lr

from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'fixed', FixedPolicy)