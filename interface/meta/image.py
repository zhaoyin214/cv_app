#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   image.py
@time    :   2020/05/29 15:54:21
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   the interface of images
"""

__author__ = "XiaoY"


import numpy as np
from abc import ABCMeta, abstractproperty
from typing import List

Image = np.array

Blob = np.array

class ImageSize(metaclass=ABCMeta):

    @abstractproperty
    def width(self) -> int:
        return self._image.shape[1]

    @abstractproperty
    def height(self) -> int:
        return self._image.shape[0]

    @abstractproperty
    def image(self) -> Image:
        return self._image
    @image.setter
    def image(self, image: Image) -> None:
        self._image = image
