#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   visualizer.py
@time    :   2020/05/21 14:55:49
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"

from interface.meta import IConfBoxIterator, IPointIterator
import cv2
import numpy as np
from typing import Dict, List, Text

class Plotter(object):

    def __init__(
        self,
        font_scale: float=0.5,
        font_face: int=cv2.FONT_HERSHEY_SIMPLEX,
        font_color: List[int]=(0, 255, 255),
        font_thickness: int=1,
        line_type: int=cv2.LINE_AA,
        line_thickness: int=2,
        line_color: List[int]=(0, 255, 0),
        point_size: int=1,
        point_color: List[int]=(0, 255, 255)
    ) -> None:
        self._font_scale = font_scale
        self._font_face = font_face
        self._font_color = font_color
        self._font_thickness = font_thickness
        self._line_type = line_type
        self._line_thickness = line_thickness
        self._line_color = line_color
        self._pt_size = point_size
        self._pt_color = point_color

class BBoxPlottor(Plotter):

    def _plot(self, image: np.array, bboxes: IConfBoxIterator) -> np.array:

        image = image.copy()
        for bbox in bboxes:

            cv2.rectangle(
                img=image, pt1=(bbox.xmin, bbox.ymin), pt2=(bbox.xmax, bbox.ymax),
                color=self._line_color,
                thickness=self._line_thickness,
                lineType=self._line_type
            )
            cv2.putText(
                img=image, text="{:.3f}".format(bbox.conf),
                org=(bbox.xmin, bbox.ymin),
                fontFace=self._font_face,
                fontScale=self._font_scale,
                color=self._font_color,
                thickness=self._font_thickness,
                lineType=self._line_type
            )

        return image

    def __call__(self, image: np.array, bboxes: IConfBoxIterator) -> np.array:
        return self._plot(image, bboxes)


class KeyPointPlotter(Plotter):

    def _plot(self, image: np.array, keypoints: IPointIterator) -> np.array:

        image = image.copy()
        for pt in keypoints:

            cv2.circle(
                img=image,
                center=(int(pt.x), int(pt.y)),
                radius=self._pt_size,
                color=self._pt_color,
                thickness=-1,
                lineType=cv2.FILLED
            )
            # cv2.putText(
            #     img=image,
            #     text="{}".format(i),
            #     org=(int(pt[0]), int(pt[1])),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.5,
            #     color=(0, 0, 255),
            #     thickness=1,
            #     lineType=cv2.LINE_AA
            # )

        return image

    def __call__(self, image: np.array, keypoints: IPointIterator) -> np.array:
        return self._plot(image, keypoints)


def _show(image: np.array, is_show: bool) -> None:

    if is_show:
        cv2.imshow("det", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
        cv2.destroyAllWindows()

def _save(image: np.array, output_path: Text) -> None:

    if output_path:
        cv2.imwrite(output_path, image)

def _plot_bboxes(
    image: np.array, bboxes: IConfBoxIterator
) -> np.array:

    bbox_plottor = BBoxPlottor()
    return bbox_plottor(image, bboxes)

def _plot_keypoints(
    image: np.array, keypoints: IPointIterator
) -> np.array:

    pt_plottor = KeyPointPlotter()
    return pt_plottor(image, keypoints)

def _plot_multi_obj_alignment(
    image: np.array,
    bboxes: IConfBoxIterator,
    kpts_aggregate: List[IPointIterator]
) -> np.array:

    bbox_plottor = BBoxPlottor()
    pt_plottor = KeyPointPlotter()

    image = bbox_plottor(image, bboxes)
    for kpts in kpts_aggregate:
        image = pt_plottor(image, kpts)

    return image

def show_bboxes(
    image: np.array, bboxes: IConfBoxIterator,
    output_path: Text=None, is_show: bool=True
) -> None:

    image = _plot_bboxes(image, bboxes)
    _show(image, is_show)
    _save(image, output_path)

def show_keypoints(
    image: np.array, keypoints: IPointIterator,
    output_path: Text=None, is_show: bool=True
) -> None:

    image = _plot_keypoints(image, keypoints)
    _show(image, is_show)
    _save(image, output_path)

def show_multi_obj_alignment(
    image: np.array,
    bboxes: IConfBoxIterator,
    kpts_aggregate: List[IPointIterator],
    output_path: Text=None, is_show: bool=True
) -> None:

    image = _plot_multi_obj_alignment(image, bboxes, kpts_aggregate)

    _show(image, is_show)
    _save(image, output_path)

