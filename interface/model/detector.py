#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   detector.py
@time    :   2020/05/26 11:36:09
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of detectors,
             which output bounding boxes given an image as input
"""

__author__ = "XiaoY"

import numpy as np
from .base import IBaseModel
from interface.meta import IConfBoxIterator
from abc import abstractmethod


class IBaseDetector(IBaseModel):

    @abstractmethod
    def __call__(self, image: np.array) -> IConfBoxIterator:
        pass
