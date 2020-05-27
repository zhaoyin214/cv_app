#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   bbox.py
@time    :   2020/05/26 11:55:04
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   concrete box classes
"""

__author__ = "XiaoY"

from interface.meta import IBox, IConfBox
from typing import List

class Box(IBox):
    """
    a box
    """
    def __init__(
        self, xmin: int, ymin: int, xmax:int, ymax: int
    ) -> None:
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def xmin(self) -> int:
        return self._xmin
    @xmin.setter
    def xmin(self, value: int) -> None:
        self._xmin = value

    @property
    def ymin(self) -> int:
        return self._ymin
    @ymin.setter
    def ymin(self, value: int) -> None:
        self._ymin = value

    @property
    def xmax(self) -> int:
        return self._xmax
    @xmax.setter
    def xmax(self, value: int) -> None:
        self._xmax = value

    @property
    def ymax(self) -> int:
        return self._ymax
    @ymax.setter
    def ymax(self, value: int) -> None:
        self._ymax = value

    @property
    def width(self) -> int:
        return self._xmax - self._xmin + 1

    @property
    def height(self) -> int:
        return self._ymax - self._ymin + 1

class ConfBox(IConfBox, Box):
    """
    a box with confidence
    """
    def __init__(
        self, conf: float, xmin: int, ymin: int, xmax:int, ymax: int
    ) -> None:
        super(ConfBox, self).__init__(xmin, ymin, xmax, ymax)
        self._conf = conf

    @property
    def conf(self) -> float:
        return self._conf
    @conf.setter
    def conf(self, value: float) -> None:
        self._conf = value
