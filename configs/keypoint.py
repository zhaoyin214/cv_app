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

FACE_ALIGN_68_KP = {
    "face": {
        "mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        "right_brow": [17, 18, 19, 20, 21],
        "left_brow": [22, 23, 24, 25, 26],
        "right_eye": [36, 37, 38, 39, 40, 41],
        "left_eye": [42, 43, 44, 45, 46, 47],
        "nose": [27, 28, 29, 30, 31, 32, 33, 34, 35]
    },
    "jaw": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
}

# fan: face alignment net
FACE_ALIGN_68_KP_FAN = {
    "proto": os.path.join(MODEL_DIR, "fan/fan.xml"),
    "weights": os.path.join(MODEL_DIR, "fan/fan.bin"),
    "backend": "DLDT",
    "input_height": 256,
    "input_width": 256,
    "swap_rb": True,
    "crop": False,
    "mean": (0, 0, 0),
    "scale_factor": 1 / 255,
    "num_kpts": 68,
    "heat_map_indices": list(range(68)),
    "threshold": 0.2,
    # "decoding_mode": "hard", # soft or hard
    "decoding_mode": "soft", # soft or hard
}

FACE_ALIGN_70_KP_OPENPOSE = {
    "proto": os.path.join(MODEL_DIR, "openpose/face/pose_deploy.prototxt"),
    "weights": os.path.join(MODEL_DIR, "openpose/face/pose_iter_116000.caffemodel"),
    "backend": "Caffe",
    "input_height": 368,
    "input_width": 368,
    "swap_rb": False,
    "crop": False,
    "mean": (0, 0, 0),
    "scale_factor": 1 / 255,
    "heat_map_indices": list(range(70)),
    "threshold": 0.2,
    "decoding_mode": "hard", # soft or hard
}

# intel facial-landmarks-35-adas-0002
# blob shape: [1, 70]
# [x0, y0, x1, y1, ..., x34, y34]
FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32 = {
    "proto": os.path.join(MODEL_DIR, "intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"),
    "weights": os.path.join(MODEL_DIR, "intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin"),
    "backend": "DLDT",
    "input_height": 60,
    "input_width": 60,
    "swap_rb": False,
    "crop": False,
    "num_kpts": 35,
}

# dlib shape predictor
FACE_ALIGN_68_KP_DLIB = {
    "model": os.path.join(MODEL_DIR, "dlib/shape_predictor_68_face_landmarks.dat"),
    "num_kpts": 68,
}
