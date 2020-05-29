#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   det_net.py
@time    :   2020/05/27 12:40:05
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   object detection net
"""

__author__ = "XiaoY"

from interface.meta import IConfBoxIterator, Image, Blob
from interface.model import IBaseDetector
from meta import Aggregate, ConfBox
from .base_net import ClassNetCV
from typing import Dict

class DetectNetCV(IBaseDetector, ClassNetCV):
    """
    the net for object detection
    """

    def __init__(self, config: Dict):
        super(DetectNetCV, self).__init__(config)
        self._bbox = config.get("bbox")

    def _post_proc(
        self, image: Image, pred: Blob
    ) -> IConfBoxIterator:

        image_height, image_width = image.shape[0 : 2]
        # bounding boxes
        bboxes = Aggregate()
        for i in range(pred.shape[2]):

            # confidence map of corresponding hand's part.
            conf = pred[0, 0, i, self._bbox["conf"]]
            if conf > self._threshold:
                x_min = int(pred[0, 0, i, self._bbox["xmin"]] * image_width)
                y_min = int(pred[0, 0, i, self._bbox["ymin"]] * image_height)
                x_max = int(pred[0, 0, i, self._bbox["xmax"]] * image_width)
                y_max = int(pred[0, 0, i, self._bbox["ymax"]] * image_height)

                bboxes.append(ConfBox(conf, x_min, y_min, x_max, y_max))

        return bboxes

    def __call__(self, image: Image) -> IConfBoxIterator:
        return self._call(image)
