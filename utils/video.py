#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   video.py
@time    :   2020/05/21 19:16:23
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"


import cv2
import numpy as np
from typing import Text, Tuple

class VideoReader(object):
    """
    video reader
    """
    def __init__(self, filepath: Text) -> None:
        try:
            self._cap = cv2.VideoCapture(filepath)
        except Exception as e:
            raise e

        self._fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self._size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    @property
    def size(self):
        return self._size
    @property
    def fps(self):
        return self._fps

    def read(self) -> np.array:

        while True:

            is_grabbed, image = self._cap.read()

            if not is_grabbed:
                break

            yield image

    def release(self) -> None:
        self._cap.release()


class VideoWriter(object):
    """
    video writer
    """

    def __init__(
        self, output_path: Text, size: Tuple[int], fps: int,
        format: Text="MJPG",
    ) -> None:
        self._writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*format),
            fps, size
        )

    def write(self, image: np.array) -> None:
        self._writer.write(image)

    def release(self) -> None:
        self._writer.release()