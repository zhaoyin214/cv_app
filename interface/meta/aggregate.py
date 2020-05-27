#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   aggregate.py
@time    :   2020/05/22 16:44:57
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   an interface set of iterators
"""

__author__ = "XiaoY"

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List

#----------------#
#-- interfaces --#
#----------------#
Iterator = List

class IAggregate(metaclass=ABCMeta):

    @abstractmethod
    def append(self, obj: Any) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self) -> Any:
        pass

    @abstractproperty
    def iterator(self) -> Iterator:
        pass

