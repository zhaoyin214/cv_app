#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   demo_face_detection_video.py
@time    :   2020/05/21 20:40:51
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   face detection for video
"""

__author__ = "XiaoY"


from factory import face_detector_factory, face_alignment_factory
from app import BBoxKeypointApp, FaceSwapApp
from utils.display.visualizer import _plot_multi_obj_alignment
from utils.display.video import VideoReader, VideoWriter
from configs.data_path import VIDEO_DIR
from configs.face_swap import DELAUNAY_TRI_FACE_ALIGN_68_KP, CONTOUR_FACE_ALIGN_68_KP
import cv2
import os

if __name__ == "__main__":
    user_image = cv2.imread("./data/face_swap/hillary_clinton.jpg")
    output_path = "./output/face_swap_out_01.avi"
    # face detection
    face_detector_key = "FACE_DET_INTEL_RETAIL_0005_FP32"
    # face alignment
    face_aligner_key = "FACE_ALIGN_68_KP_FAN"
    video_reader = VideoReader(os.path.join(VIDEO_DIR, "The_Marvel_Bunch_crop.mp4"))
    video_writer = VideoWriter(
        output_path, video_reader.size, video_reader.fps
    )
    face_detector = face_detector_factory[face_detector_key]
    face_aligner = face_alignment_factory[face_aligner_key]
    bbox_kp_app = BBoxKeypointApp(face_detector, face_aligner, padding=0.3)
    face_swap_app = FaceSwapApp(
        DELAUNAY_TRI_FACE_ALIGN_68_KP,
        CONTOUR_FACE_ALIGN_68_KP,
        bbox_kp_app
    )
    face_swap_app.user = user_image
    for image in video_reader.read():

        image = face_swap_app(image)
        cv2.imshow("swap", image)
        cv2.waitKey(1)
        video_writer.write(image)

    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
