#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   factory.py
@time    :   2020/05/29 11:46:27
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"


from ..model import IBaseModel
from abc import ABCMeta, abstractmethod
from typing import Dict

class IModelFactory(metaclass=ABCMeta):

    @abstractmethod
    def create_model(self, config: Dict) -> IBaseModel:
        pass