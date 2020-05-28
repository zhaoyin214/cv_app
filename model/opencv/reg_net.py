#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   reg_net.py
@time    :   2020/05/28 14:13:07
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"


from interface.meta import IPointIterator
from interface.model import IBaseAlignment
from interface.meta import Image
from meta import Aggregate, Point
from .base_net import BaseNetCV
from typing import Dict

class AlignmentNetCV(IBaseAlignment, BaseNetCV):
    """
    the concrete alignment class, which output a set of keypoints given an image

    return

    [point, ...]
    """
    def __init__(self, config: Dict) -> None:
        super(AlignmentNetCV, self).__init__(config)
        self._num_kpts = config.get("num_kpts")

    def _post_proc(
        self, image: Image, pred: IPointIterator
    ) -> IPointIterator:

        image_height, image_width = image.shape[0 : 2]

        keypoints = Aggregate()
        for idx_pt in range(self._num_kpts):
            pt = Point(
                x=int(image_width * pred[0, 2 * idx_pt]),
                y=int(image_height * pred[0, 2 * idx_pt + 1])
            )
            keypoints.append(pt)

        return keypoints

    def __call__(self, image: Image) -> IPointIterator:
        return self._call(image)
