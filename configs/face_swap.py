import json
import os

config = "./data/face_swap/delaunay_face_align_68_kp.json"
if os.path.isfile(config):
    with open(config, mode="r", encoding="utf-8") as f:
        DELAUNAY_TRI_FACE_ALIGN_68_KP, CONTOUR_FACE_ALIGN_68_KP = \
            json.load(f)
else:
    from utils.warp import DelaunayTrianglation
    filepath = "./data/face_swap/hillary_clinton.jpg.txt"
    delaunay = DelaunayTrianglation(filepath)
    DELAUNAY_TRI_FACE_ALIGN_68_KP = delaunay.trianglar_grid
    CONTOUR_FACE_ALIGN_68_KP = delaunay.contour
    with open(config, mode="w", encoding="utf-8") as f:
        json.dump(
            [DELAUNAY_TRI_FACE_ALIGN_68_KP, CONTOUR_FACE_ALIGN_68_KP],
            fp=f
        )
