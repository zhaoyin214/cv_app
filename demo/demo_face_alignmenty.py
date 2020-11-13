from factory import face_detector_factory, face_alignment_factory
from app import BBoxKeypointApp
from utils.display.visualizer import show_multi_obj_alignment
import cv2
import os


if __name__ == "__main__":

    output_path = "./output/face_alignment_out.jpg"
    image = cv2.imread(filename="./img/hoffman.jpg")
    # image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")

    # face detection
    face_detector_key = "FACE_DET_INTEL_RETAIL_0005_FP32"

    # face alignment
    # face_aligner_key = "FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32"
    # face_aligner_key = "FACE_ALIGN_68_KP_DLIB"
    # face_aligner_key = "FACE_ALIGN_70_KP_OPENPOSE"
    face_aligner_key = "FACE_ALIGN_68_KP_FAN"

    face_detector = face_detector_factory[face_detector_key]
    face_aligner = face_alignment_factory[face_aligner_key]

    app = BBoxKeypointApp(face_detector, face_aligner)

    bboxes, kpts_aggregate = app(image)

    show_multi_obj_alignment(image, bboxes, kpts_aggregate, output_path)
