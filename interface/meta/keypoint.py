#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   keypoint.py
@time    :   2020/05/25 16:31:41
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of keypoints
"""

__author__ = "XiaoY"

from .aggregate import Iterator
from abc import ABCMeta, abstractmethod, abstractproperty

class IPoint(metaclass=ABCMeta):
    """
    a basic point
    """

    @abstractproperty
    def x(self) -> int:
        pass
    @x.setter
    def x(self, value: int) -> None:
        pass

    @abstractproperty
    def y(self) -> int:
        pass
    @y.setter
    def y(self, value: int) -> None:
        pass

class IConfPoint(IPoint):
    """
    a point with confidence
    """

    @abstractproperty
    def conf(self) -> int:
        pass
    @conf.setter
    def conf(self, value: float) -> None:
        pass

IPointIterator = Iterator[IPoint]
