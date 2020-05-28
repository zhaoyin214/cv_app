#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   bbox_keypoint.py
@time    :   2020/05/28 16:46:03
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of the application with object detection and
             keypoints localization tasks
"""

__author__ = "XiaoY"

from ..meta import Image, IBoxIterator, IPointIterator
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

IBBoxKeypointIterator = Tuple[IBoxIterator, List[IPointIterator]]

class IBBoxKeypointApp(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, image: Image) -> IBBoxKeypointIterator:
        pass
