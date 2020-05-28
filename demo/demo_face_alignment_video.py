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


from model.opencv import DetectNetCV, AlignmentNetCV
from app import BBoxKeypointApp
from utils.display.visualizer import _plot_multi_obj_alignment
from utils.display.video import VideoReader, VideoWriter
from configs.data_path import VIDEO_DIR
from configs.bbox import FACE_DET_INTEL_RETAIL_0005_FP32
from configs.keypoint import FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32
import cv2
import os


if __name__ == "__main__":
    output_path = "./output/face_alignment_out_01.avi"

    video_reader = VideoReader(os.path.join(VIDEO_DIR, "The_Marvel_Bunch.mp4"))
    video_writer = VideoWriter(
        output_path, video_reader.size, video_reader.fps
    )
    face_detector = DetectNetCV(config=FACE_DET_INTEL_RETAIL_0005_FP32)
    face_aligner = AlignmentNetCV(config=FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32)
    app = BBoxKeypointApp(face_detector, face_aligner)

    for image in video_reader.read():

        # cv2.imshow("ori", image)
        bboxes, kpts_aggregate = app(image)
        image = _plot_multi_obj_alignment(image, bboxes, kpts_aggregate)
        cv2.imshow("align", image)
        cv2.waitKey(1)
        video_writer.write(image)

    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()