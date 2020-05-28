from model.opencv import DetectNetCV, AlignmentNetCV
from app import BBoxKeypointApp
from utils.display.visualizer import show_multi_obj_alignment
from configs.bbox import FACE_DET_INTEL_RETAIL_0005_FP32
from configs.keypoint import FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32
import cv2
import os


if __name__ == "__main__":

    output_path = "./output/face_alignment_out.jpg"
    image = cv2.imread(filename="./img/hoffman.jpg")
    # image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")

    face_detector = DetectNetCV(config=FACE_DET_INTEL_RETAIL_0005_FP32)
    face_aligner = AlignmentNetCV(config=FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32)

    app = BBoxKeypointApp(face_detector, face_aligner)

    bboxes, kpts_aggregate = app(image)

    show_multi_obj_alignment(image, bboxes, kpts_aggregate, output_path)
