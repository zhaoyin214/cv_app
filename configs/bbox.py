#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   bbox.py
@time    :   2020/05/27 13:01:16
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   the config of object detection tasks
"""

__author__ = "XiaoY"

from .data_path import MODEL_DIR
import os

# intel face-detection-retail-0005
# blob shape: [1, 1, N, 7]
# where N is the number of detected bounding boxes
# [image_id, label, conf, x_min, y_min, x_max, y_max]
FACE_DET_INTEL_RETAIL_0005_FP32 = {
    "proto": os.path.join(MODEL_DIR, "intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"),
    "weights": os.path.join(MODEL_DIR, "intel/face-detection-retail-0005/FP32/face-detection-retail-0005.bin"),
    "backend": "DLDT",
    # "input_height": 300,
    # "input_width": 300,
    "swap_rb": False,
    "crop": False,
    "bbox": {"conf": 2, "xmin": 3, "ymin": 4, "xmax": 5, "ymax": 6},
    "threshold": 0.88
}
