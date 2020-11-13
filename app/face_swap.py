#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   face_swap.py
@time    :   2020/09/28 17:16:57
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"

import cv2
import numpy as np

from .bbox_keypoint import BBoxKeypointApp
from interface.meta import Image, IConfBoxIterator, IPointIterator
from interface.app import IBBoxKeypointApp
from typing import Dict, List

CV_RECT_XMIN = 0
CV_RECT_YMIN = 1
CV_RECT_WIDTH = 2
CV_RECT_HEIGHT = 3

MASK_VALID_FLOAT = (1, 1, 1)
MASK_VALID_UBYTE = (255, 255, 255)

class FaceSwap(object):
    """
    face swap with delaunay triangular
        - clone mode: cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    attributes:
        keypoints (Aggregate)
        delaunay_triangulation
        contour (1-dim array)
    """
    _seamless_mode = [
        cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    ]
    def __init__(
        self,
        trianglar_grid,
        contour
    ):
        """
        constructor
        """
        self._trianglar_grid = trianglar_grid
        self._face_contour = contour

    def _warp(
        self,
        image_src: Image,
        image_dst: Image,
        points_src: IPointIterator,
        points_dst: IPointIterator
    ) -> Image:
        """
        affine transform with delaunay triangulation
        """
        image_dst_copy = image_dst.copy().astype(np.float32)
        # traversing triangules
        for tri in self._trianglar_grid:
            tri_src = []
            tri_dst = []
            for pt in tri:
                tri_src.append([points_src[pt].x, points_src[pt].y])
                tri_dst.append([points_dst[pt].x, points_dst[pt].y])
            image_dst_copy = self._warp_affine_triangule(
                image_src=image_src,
                image_dst=image_dst_copy,
                tri_src=tri_src,
                tri_dst=tri_dst
            )
        # clamp
        image_dst_copy = np.clip(a=image_dst_copy, a_min=0, a_max=255)
        image_dst_copy = image_dst_copy.astype(np.uint8)
        return image_dst_copy

    def _warp_affine_triangule(
        self,
        image_src: Image,
        image_dst: Image,
        tri_src: List[List],
        tri_dst: List[List],
        alpha: float=1
    ) -> Image:
        """
        affine transform for each triangule
        arguments:
            image_src - source
            image_dst - destination
            tri_src - vertices of a triangule
            tri_dst - vertices of a triangule
        return:
            image_copy - warpped image
        """
        tri_src = np.array(tri_src)
        tri_dst = np.array(tri_dst)
        rect_src = cv2.boundingRect(tri_src)
        rect_dst = cv2.boundingRect(tri_dst)
        # roi - "ymin", "xmin", "ymax", "xmax"
        roi_src = [
            rect_src[CV_RECT_YMIN],
            rect_src[CV_RECT_XMIN],
            rect_src[CV_RECT_YMIN] + rect_src[CV_RECT_HEIGHT],
            rect_src[CV_RECT_XMIN] + rect_src[CV_RECT_WIDTH]
        ]
        roi_dst = [
            rect_dst[CV_RECT_YMIN],
            rect_dst[CV_RECT_XMIN],
            rect_dst[CV_RECT_YMIN] + rect_dst[CV_RECT_HEIGHT],
            rect_dst[CV_RECT_XMIN] + rect_dst[CV_RECT_WIDTH]
        ]
        # offsets to roi origin
        tri_src[:, 0] -= rect_src[CV_RECT_XMIN]
        tri_src[:, 1] -= rect_src[CV_RECT_YMIN]
        tri_dst[:, 0] -= rect_dst[CV_RECT_XMIN]
        tri_dst[:, 1] -= rect_dst[CV_RECT_YMIN]
        image_src_roi = image_src[
            roi_src[0] : roi_src[2], roi_src[1] : roi_src[3], ...
        ]
        # affine matrix
        affine_matrix = cv2.getAffineTransform(
            src=tri_src.astype(np.float32),
            dst=tri_dst.astype(np.float32)
        )
        # affine transforming the source image
        # size w x h
        size = (rect_dst[CV_RECT_WIDTH], rect_dst[CV_RECT_HEIGHT])
        image_wrap = cv2.warpAffine(
            src=image_src_roi,
            M=affine_matrix,
            dsize=size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        # mask of the destination triangule
        mask = np.zeros(
            shape=(
                rect_dst[CV_RECT_HEIGHT],
                rect_dst[CV_RECT_WIDTH],
                image_src.shape[2]
            ),
            dtype=np.float32
        )
        cv2.fillConvexPoly(
            img=mask, points=tri_dst, color=MASK_VALID_FLOAT
        )
        mask *= alpha
        image_dst[roi_dst[0] : roi_dst[2], roi_dst[1] : roi_dst[3]] = \
            mask * image_wrap + \
            (1 - mask) * image_dst[roi_dst[0] : roi_dst[2], roi_dst[1] : roi_dst[3]]
        return image_dst

    def _clone(
        self,
        image_src: Image,
        image_dst: Image,
        points: IPointIterator,
        mode=cv2.NORMAL_CLONE
    ):
        """
        Poisson Image Editing
            P. Rez, M. Gangnet, A. Blake
            Acm Transactions on Graphics (2003)
        arguments:
        return:
        """
        contour_points = np.array(
            [(points[pt].x, points[pt].y) for pt in self._face_contour]
        )
        points = np.array([(pt.x, pt.y) for pt in points])
        mask = np.zeros(shape=image_dst.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            img=mask, points=contour_points, color=MASK_VALID_UBYTE
        )
        rect = cv2.boundingRect(points)
        center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        if mode in self._seamless_mode:
            image_clone = cv2.seamlessClone(
                src=image_src,
                dst=image_dst,
                mask=mask,
                p=center,
                flags=mode
            )
        else:
            raise AssertionError("ERROR: invalid clone mode!")
        return image_clone

    def __call__(
        self,
        image_src: Image,
        image_dst: Image,
        points_src: IPointIterator,
        points_dst: IPointIterator,
    ) -> Image:
        image_face_warp = self._warp(
            image_src, image_dst, points_src, points_dst
        )
        image_face_clone = self._clone(
            image_face_warp, image_dst, points_dst
        )
        return image_face_clone

class FaceSwapApp(object):
    def __init__(
        self,
        trianglar_grid,
        contour,
        bbox_kp_app: IBBoxKeypointApp
    ):
        self._face_swap = FaceSwap(trianglar_grid, contour)
        self._bbox_kp_app = bbox_kp_app
        self._user = None

    def __call__(self, image: Image) -> Image:
        kpts = self._face_det_align(image)
        if kpts:
            return self._face_swap(
                self._user["image"],
                image,
                self._user["keypoints"],
                kpts
            )
        else:
            return image

    def _face_det_align(self, image: Image) -> IPointIterator:
        bboxes, kpts = self._bbox_kp_app(image)
        assert len(bboxes) <= 1, "Error: Too Many Faces!"
        if len(bboxes) == 1:
            return kpts[0]
        else:
            return None

    def _set_user(self, image: Image):
        kpts = self._face_det_align(image)
        assert kpts is not None, "Error: No Face Found Error!"
        self._user = {
            "image" : image,
            "keypoints": kpts
        }

    @property
    def user(self) -> None:
        return self._user
    @user.setter
    def user(self, image: Image) -> Dict:
        self._set_user(image)
