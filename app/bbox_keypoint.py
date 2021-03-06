#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   bbox_keypoint.py
@time    :   2020/05/22 16:17:46
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"

from interface.app import IBBoxKeypointIterator, IBBoxKeypointApp
from interface.model import IBaseAlignment, IBaseDetector
from interface.meta import IBox, IBoxIterator, IPointIterator, Image
from meta import Box
from utils.bbox.roi import box_padding, convert_kpts_2_global

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple

class BBoxKeypointApp(IBBoxKeypointApp):

    def __init__(
        self,
        bbox_net: IBaseDetector,
        keypoint_net: IBaseAlignment,
        padding: float=0.1
    ) -> None:
        self._bbox_net = bbox_net
        self._kpt_net = keypoint_net
        self._padding = padding

    def _predict(
        self, image: Image
    ) -> IBBoxKeypointIterator:
        bboxes = self._bbox_net(image)
        kpts_aggregate = []
        for bbox in bboxes:
            kpts_aggregate.append(self._predict_obj_kps(image, bbox))
        return bboxes, kpts_aggregate

    def _predict_obj_kps(
        self, image: Image, bbox: IBox
    ) -> IPointIterator:
        border = Box(0, 0, image.shape[1], image.shape[0])
        roi = box_padding(bbox, border, self._padding)
        image_crop = image[
            roi.ymin : roi.ymax, roi.xmin : roi.xmax, :
        ]
        keypoints = self._kpt_net(image_crop)
        keypoints = convert_kpts_2_global(keypoints, roi)
        return keypoints

    def __call__(
        self, image: Image
    ) -> IBBoxKeypointIterator:
        return self._predict(image)

class BBoxKeypointBaggingApp(BBoxKeypointApp):

    def _predict_obj_kps(
        self, image: Image, bbox: IBox
    ) -> IPointIterator:
        pass