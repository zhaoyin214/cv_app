#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   reg_net.py
@time    :   2020/05/28 12:49:36
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   the face alignment with dlib.shape_predictor()
"""

__author__ = "XiaoY"


from interface.meta import IPointIterator
from interface.model import IBaseModel
from interface.meta import Image
import dlib
from typing import Dict

class RegNetDlib(IBaseModel):
    """
    the adaptor of dlib.shape_predictor()
    """

    def __init__(self, config: Dict) -> None:

        self._predictor = None
        try:
            self._predictor = dlib.shape_predictor(config["model"])
        except Exception as e:
            raise e

    def _predict(self, image: Image) -> IPointIterator:
        landmarks = self._predictor(image)
        pass

    def __call__(self, image: Image) -> IPointIterator:
        pass
