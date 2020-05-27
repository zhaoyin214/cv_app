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

from interface.meta import IConfBoxIterator
from interface.model import IBaseDetector
from meta import Aggregate, ConfBox
from .base_net import ClassNetCV
import numpy as np
from typing import Dict

class DetectNetCV(IBaseDetector, ClassNetCV):
    """
    the net for object detection
    """

    def __init__(self, config: Dict):
        super(DetectNetCV, self).__init__(config)
        self._bbox = config.get("bbox")

    def _post_proc(
        self, image: np.array, pred: np.array
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

    def __call__(self, image: np.array) -> IConfBoxIterator:
        blob = self._input_blob(image)
        output = self._predict(blob)
        bboxes = self._post_proc(image, output)

        return bboxes

