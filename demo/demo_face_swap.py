from factory import face_detector_factory, face_alignment_factory
from app import BBoxKeypointApp, FaceSwapApp
import cv2
import os

from configs.face_swap import DELAUNAY_TRI_FACE_ALIGN_68_KP, CONTOUR_FACE_ALIGN_68_KP

if __name__ == "__main__":

    output_path = "./output/face_swap_out.jpg"
    image_src = cv2.imread(filename="./data/face_swap/hillary_clinton.jpg")
    image_dst = cv2.imread(filename="./img/narendra_modi.jpg")
    # face detection
    face_detector_key = "FACE_DET_INTEL_RETAIL_0005_FP32"
    face_detector = face_detector_factory[face_detector_key]
    # face alignment
    face_aligner_key = "FACE_ALIGN_68_KP_FAN"
    face_aligner = face_alignment_factory[face_aligner_key]
    face_det_align_app = BBoxKeypointApp(face_detector, face_aligner)
    # face swap
    face_swap = FaceSwapApp(
        trianglar_grid=DELAUNAY_TRI_FACE_ALIGN_68_KP,
        contour=CONTOUR_FACE_ALIGN_68_KP,
        bbox_kp_app=face_det_align_app
    )

    face_swap.user = image_src
    image_dst = face_swap(image_dst)
    cv2.imshow("show", image_dst)
    cv2.waitKey(0)
    cv2.destroyWindow("show")
