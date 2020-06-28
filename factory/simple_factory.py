from model.dlib import DlibShapePredictor
from model.opencv import DetectNetCV, AlignmentNetCV, AlignmentHeatMapNetCV

from configs.bbox import FACE_DET_INTEL_RETAIL_0005_FP32


from configs.keypoint import \
    FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32, \
    FACE_ALIGN_70_KP_OPENPOSE, \
    FACE_ALIGN_68_KP_DLIB

intel_detector = {
    "FACE_DET_INTEL_RETAIL_0005_FP32": FACE_DET_INTEL_RETAIL_0005_FP32,
}
intel_alignment = {
    "FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32": FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32,
}
intel_alignment_heat_map = {
    "FACE_ALIGN_70_KP_OPENPOSE": FACE_ALIGN_70_KP_OPENPOSE,
}
dlib_alignmen = {
    "FACE_ALIGN_68_KP_DLIB": FACE_ALIGN_68_KP_DLIB
}

face_detector_factory = {
    key: DetectNetCV(value) for key, value in intel_detector.items()
}

face_alignment_factory = {
    key: AlignmentNetCV(value) for key, value in intel_alignment.items()
}
face_alignment_factory.update({
    key: AlignmentHeatMapNetCV(value) for key, value in intel_alignment_heat_map.items()
})
face_alignment_factory.update({
    key: DlibShapePredictor(value) for key, value in dlib_alignmen.items()
})