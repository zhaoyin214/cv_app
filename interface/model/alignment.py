#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   alignment.py
@time    :   2020/05/28 12:13:14
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of alignment tasks,
             which output keypoints, [[x, y], ...] given an image as input
"""

__author__ = "XiaoY"


from interface.meta import IPointIterator, Image
from .base import IBaseModel
from abc import abstractmethod

class IBaseAlignment(IBaseModel):
    @abstractmethod
    def __call__(self, image: Image) -> IPointIterator:
        """
        return

        keypoints: [point, ...]
        """
        pass
