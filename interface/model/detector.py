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

from interface.meta import IConfBoxIterator
from interface.meta import Image
from .base import IBaseModel
import numpy as np
from abc import abstractmethod


class IBaseDetector(IBaseModel):
    """
    the interface of the base detector
    """
    @abstractmethod
    def __call__(self, image: Image) -> IConfBoxIterator:
        """
        return

        bounding boxes, [conf_box, ...]
        """
        pass
