import cv2

from openvino.inference_engine import IENetwork

net = IENetwork("./data/models/intel/facial-landmarks-35-adas-0002/facial-landmarks-35-adas-0002.xml")

print(net)