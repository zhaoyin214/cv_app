from model.opencv import AlignmentNetCV
from configs.keypoint import FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32
from utils.display.visualizer import show_keypoints
import cv2

if __name__ == "__main__":
    output_path = "./output/face_align_out.jpg"
    image = cv2.imread(filename="./img/49.jpg")

    face_alignment = AlignmentNetCV(
        config=FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32
    )

    keypoints = face_alignment(image)
    show_keypoints(image, keypoints, output_path)
