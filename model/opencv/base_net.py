#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   base_net.py
@time    :   2020/05/26 16:03:21
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   base net classes with opencv.dnn as backbone
"""

__author__ = "XiaoY"

from interface.model import IBaseModel
from interface.meta import Image, Blob
import cv2
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

class BaseNetCV(IBaseModel):
    """
    the base net class for all cv2.dnn nets
    """

    _support_backends = [
        "Caffe", "TensorFlow", "Darknet", "DLDT"
    ]

    def __init__(self, config: Dict) -> None:

        backend = config["backend"]
        if backend not in self._support_backends:
            raise ValueError(
                "Backend {} is not supported.".format(backend)
            )
        try:
            self._net = cv2.dnn.readNet(
                model=config["weights"],
                config=config["proto"],
                framework=backend
            )
        except Exception as e:
            raise e

        self._input_height = config.get("input_height")
        self._input_width = config.get("input_width")
        self._swap_rb = config.get("swap_rb", False)
        self._crop = config.get("crop", False)
        self._mean = config.get("mean", (0, 0, 0))
        self._scale_factor = config.get("scale_factor", 1)
        self._is_gray = config.get("gray", False)

    def _predict(self, blob: Blob) -> Any:
        self._net.setInput(blob)
        output = self._net.forward()
        return output

    def _input_blob(self, image: Image) -> Blob:
        if self._is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width = image.shape[0 : 2]
        if self._input_height and self._input_width:
            input_width = self._input_width
            input_height = self._input_height
        elif self._input_height:
            input_height = self._input_height
            aspect_ratio = image_width / image_height
            # input image dimensions for the network
            input_width = int(aspect_ratio * input_height)
        elif self._input_width:
            input_width = self._input_width
            aspect_ratio = image_width / image_height
            # input image dimensions for the network
            input_height = int(input_width / aspect_ratio)
        else:
            input_height = image_height
            input_width = image_width
        # input image dimensions for the network
        blob = cv2.dnn.blobFromImage(
            image=image, scalefactor=self._scale_factor,
            size=(input_width, input_height),
            mean=self._mean, swapRB=self._swap_rb, crop=self._crop
        )
        return blob

    def _pre_proc(self, image: Image) -> Image:
        return image

    def _post_proc(self, image: Image, pred: Blob) -> Any:
        return pred

    def _call(self, image: Image) -> Any:
        image = self._pre_proc(image)
        blob = self._input_blob(image)
        pred = self._predict(blob)
        output = self._post_proc(image, pred)
        return output

    def __call__(self, image: Image) -> Any:
        return self._call(image)

class ClassNetCV(BaseNetCV):
    """
    a base net for classification
    """
    def __init__(self, config: Dict) -> None:
        super(ClassNetCV, self).__init__(config)
        self._threshold = config.get("threshold", 0.5)

    @property
    def threshold(self) -> float:
        return self._threshold
    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self._threshold = threshold

