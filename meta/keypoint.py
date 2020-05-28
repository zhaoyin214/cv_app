#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   keypoint.py
@time    :   2020/05/28 11:22:07
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   concrete keypoint classes
"""

__author__ = "XiaoY"

from interface.meta import IPoint, IConfPoint
from typing import List

class Point(IPoint):

    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x
    @x.setter
    def x(self, value: int) -> None:
        self._x = value

    @property
    def y(self) -> int:
        return self._y
    @y.setter
    def y(self, value: int) -> None:
        self._y = value

class ConfPoint(IConfPoint, Point):

    def __init__(self, conf: float, x: int, y: int) -> None:
        super(ConfPoint, self).__init__(x, y)
        self._conf = conf

    @property
    def conf(self) -> float:
        return self._conf
    @conf.setter
    def conf(self, value: float) -> None:
        self._conf = value

