#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   keypoint.py
@time    :   2020/05/27 13:02:28
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   the config of keypoint localization tasks
"""

__author__ = "XiaoY"


from .data_path import MODEL_DIR
import os

FACE_70_KP_OPENPOSE = {
    "proto": os.path.join(MODEL_DIR, "openpose/face/pose_deploy.prototxt"),
    "weights": os.path.join(MODEL_DIR, "openpose/face/pose_iter_116000.caffemodel"),
    "backend": "Caffe",
    "input_height": 368,
    "input_width": 368,
    "swap_rb": False,
    "crop": False,
    "mean": (0, 0, 0),
    "scale_factor": 1 / 255,
    "kp_map_indices": list(range(70)),
    "threshold": 0.2
}

# intel facial-landmarks-35-adas-0002
# blob shape: [1, 70]
# [x0, y0, x1, y1, ..., x34, y34]
FACE_ALIGN_INTEL_ADAS_0002_FP32 = {
    "proto": os.path.join(MODEL_DIR, "intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"),
    "weights": os.path.join(MODEL_DIR, "intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin"),
    "backend": "DLDT",
    "input_height": 60,
    "input_width": 60,
    "swap_rb": False,
    "crop": False,
    "num_kpts": 35,
}
