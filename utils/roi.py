#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   roi.py
@time    :   2020/05/22 16:46:24
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"

from meta import IBox, IBoundingBox, Box, IKeyPoints
from copy import deepcopy

def box_padding(bbox: IBoundingBox, border: IBox, padding: float) -> IBox:

    roi = Box(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)

    padding_x = int(roi.width * padding / 2)
    padding_y = int(roi.height * padding / 2)

    roi.xmin = int(max(border.xmin, roi.xmin - padding_x))
    roi.xmax = int(min(border.xmax, roi.xmax + padding_x))
    roi.ymin = int(max(border.ymin, roi.ymin - padding_y))
    roi.ymax = int(min(border.ymax, roi.ymax + padding_y))

    return roi

def convert_kpts_2_global(kpts: IKeyPoints, roi: IBox) -> IKeyPoints:

    kpts = deepcopy(kpts)
    for pt in kpts:

        if not pt:
            continue

        pt.x = pt.x + roi.xmin - 1
        pt.y = pt.y + roi.ymin - 1

    return kpts
