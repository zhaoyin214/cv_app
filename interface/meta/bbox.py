#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   bbox.py
@time    :   2020/05/25 15:33:35
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of bounding boxes
"""

__author__ = "XiaoY"

from .aggregate import Iterator
from abc import ABCMeta, abstractmethod, abstractproperty

class IBox(metaclass=ABCMeta):
    """
    a basic box
    """

    @abstractproperty
    def xmin(self) -> int:
        pass
    @xmin.setter
    def xmin(self, value: int) -> None:
        pass

    @abstractproperty
    def ymin(self) -> int:
        pass
    @ymin.setter
    def ymin(self, value: int) -> None:
        pass

    @abstractproperty
    def xmax(self) -> int:
        pass
    @xmax.setter
    def xmax(self, value: int) -> None:
        pass

    @abstractproperty
    def ymax(self) -> int:
        pass
    @ymax.setter
    def ymax(self, value: int) -> None:
        pass

    @abstractproperty
    def width(self) -> int:
        pass

    @abstractmethod
    def height(self) -> int:
        pass

class IConfBox(IBox):
    """
    a box with confidence
    """

    @abstractproperty
    def conf(self) -> float:
        pass
    @conf.setter
    def conf(self, value: float) -> None:
        pass


IBoxIterator = Iterator[IBox]

IConfBoxIterator = Iterator[IConfBox]
