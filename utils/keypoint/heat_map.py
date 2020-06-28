#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   heat_map.py
@time    :   2020/05/29 14:48:22
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   heat map to keypoint
"""

__author__ = "XiaoY"

from interface.meta import IPoint, IPointIterator, Image
from meta import Aggregate, Point
import cv2
import numpy as np
from typing import Text

class HeatMapDecoder(object):
    """
    heat map to keypoint
    """

    def __init__(
        self, threshold: float, mode: Text="hard"
    ) -> None:
        self._mode = mode
        self._threshold = threshold
        self._decode_map = {
            "hard": self._hard_decode,
            "soft": self._soft_decode
        }

    def _hard_decode(
        self, heat_map: Image, image_width: int, image_height:int
    ) -> IPoint:
        """
        argmax
        """
        height, width = heat_map.shape[0 : 2]
        # find golbal maxima of the confidence map
        _, conf, _, kpt = cv2.minMaxLoc(src=heat_map)

        if conf > self._threshold:

            x = int(kpt[0] * image_width / width)
            y = int(kpt[1] * image_height / height)
            return Point(x, y)
        else:
            return None

    def _soft_decode(
        self, heat_map: Image, image_width: int, image_height:int
    ) -> IPoint:
        """
        soft-argmax
        """
        height, width = heat_map.shape[0 : 2]
        loc_map_x = np.tile(list(range(width)), reps=(height, 1))
        loc_map_y = np.tile(list(range(height)), reps=(width, 1)).T

        def softmax(x: Image) -> Image:
            x = (x - x.min()) / (x.max() - x.min()) * 10
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x)
        heat_map = softmax(heat_map)

        x = np.sum(loc_map_x * heat_map)
        y = np.sum(loc_map_y * heat_map)

        x = int(x * image_width / width)
        y = int(y * image_height / height)

        return Point(x, y)

    def _decode(self, heat_maps: Image, image: Image) -> IPointIterator:

        image_height, image_width = image.shape[0 : 2]

        # key points
        kpts = Aggregate()
        for i in range(heat_maps.shape[1]):
            heat_map = heat_maps[0, i, :, :]

            kpt = self._decode_map[self._mode](
                heat_map, image_width, image_height
            )
            kpts.append(kpt)

        return kpts

    def __call__(self, heat_maps: Image, image: Image) -> IPointIterator:
        return self._decode(heat_maps, image)

    @property
    def mode(self, mode: Text) -> None:
        self._mode = mode
