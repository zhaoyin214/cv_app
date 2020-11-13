#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   aggregate.py
@time    :   2020/05/27 14:32:12
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   concrete aggregate
"""

__author__ = "XiaoY"

from interface.meta import IAggregate, Iterator
from typing import Any

class Aggregate(IAggregate):

    def __init__(self):
        self._aggregate: Iterator = []

    def append(self, obj: Any) -> None:
        self._aggregate.append(obj)

    def __getitem__(self, index: int) -> Any:
        if index >= len(self._aggregate):
            raise ValueError("out of bound")

        return self._aggregate[index]

    def __len__(self) -> int:
        return len(self._aggregate)

    def __iter__(self):
        self._cnt = -1
        return self

    def __next__(self):
        self._cnt += 1
        if self._cnt < len(self._aggregate):
            return self._aggregate[self._cnt]
        else:
            raise StopIteration

    @property
    def iterator(self) -> Iterator:
        return self._aggregate

