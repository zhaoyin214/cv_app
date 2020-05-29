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
from typing import Text

class HeatMapDecoder(object):
    """
    heat map to keypoint
    """

    def __init__(
        self, num_kpts: int, threshold: float, mode: Text="hard"
    ) -> None:
        self._mode = mode
        self._num_kpts = num_kpts
        self._threshold = threshold
        self._decode_map = {
            "hard": self._hard_decode,
            "soft": self._soft_decode
        }

    def _hard_decode(
        self, heat_map: Image, image_width: int, image_height: int
    ) -> IPoint:

        heat_map = cv2.resize(
            src=heat_map, dsize=(image_width, image_height)
        )

        # find golbal maxima of the confidence map
        _, conf, _, kpt = cv2.minMaxLoc(src=heat_map)

        if conf > self._threshold:
            return Point(int(kpt[0]), int(kpt[1]))
        else:
            return None

    def _soft_decode(self, heat_map: Image) -> IPoint:
        pass

    def _decode(self, heat_maps: Image, image: Image) -> IPointIterator:

        image_height, image_width = image.shape[0 : 2]

        # key points
        kpts = Aggregate()
        for i in range(self._num_kpts):
            heat_map = heat_maps[0, i, :, :]

            kpt = self._decode_map[self._mode](
                heat_map, image_width, image_height
            )
            kpts.append(kpt)

        return kpts

    @property

