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


from interface.meta import IPointIterator, Image
from interface.model import IBaseModel
from meta import Aggregate, Point
import dlib
from typing import Dict

class DlibShapePredictor(IBaseModel):
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
        rect = dlib.rectangle(
            left=0, top=0, right=image.shape[1], bottom=image.shape[0]
        )

        keypoints = Aggregate()
        landmarks = self._predictor(image, rect)
        for idx in range(landmarks.num_parts):
            pt = Point(
                x=landmarks.part(idx).x,
                y=landmarks.part(idx).y
            )
            keypoints.append(pt)

        return keypoints

    def __call__(self, image: Image) -> IPointIterator:
        return self._predict(image)
