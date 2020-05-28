#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   demo_face_detection.py
@time    :   2020/05/21 20:40:31
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   face detection
"""

__author__ = "XiaoY"


from model.opencv import DetectNetCV
from utils.display.visualizer import show_bboxes
from configs.bbox import FACE_DET_INTEL_RETAIL_0005_FP32
import cv2


if __name__ == "__main__":
    output_path = "./output/face_det_out.jpg"
    # image = cv2.imread(filename="./img/190.jpg")
    # image = cv2.imread(filename="./img/221.jpg")
    image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")

    face_detector = DetectNetCV(config=FACE_DET_INTEL_RETAIL_0005_FP32)

    bboxes = face_detector(image)
    show_bboxes(image, bboxes, output_path)
