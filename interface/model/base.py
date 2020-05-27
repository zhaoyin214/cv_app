#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   base.py
@time    :   2020/05/26 11:28:49
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an basic interface set of model
"""

__author__ = "XiaoY"

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any

class IBaseModel(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, image: np.array) -> Any:
        pass
