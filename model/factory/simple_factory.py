from ..dlib import DlibShapePredictor
from ..opencv import DetectNetCV, AlignmentNetCV

from configs.bbox import FACE_DET_INTEL_RETAIL_0005_FP32
from configs.keypoint import FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32, FACE_ALIGN_68_KP_DLIB

intel_detector_map = {
    "FACE_DET_INTEL_RETAIL_0005_FP32": FACE_DET_INTEL_RETAIL_0005_FP32,
}
intel_alignment_map = {
    "FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32": FACE_ALIGN_35_KP_INTEL_ADAS_0002_FP32,
}
dlib_alignmen_map = {
    "FACE_ALIGN_68_KP_DLIB": FACE_ALIGN_68_KP_DLIB
}

face_detector_factory = {
    key: DetectNetCV(value) for key, value in intel_detector_map.items()
}

face_alignment_factory = {
    key: AlignmentNetCV(value) for key, value in intel_alignment_map.items()
}

face_alignment_factory.update({
    key: DlibShapePredictor(value) for key, value in dlib_alignmen_map.items()
})