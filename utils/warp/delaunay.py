import cv2
import numpy as np
import os

from typing import Text
from interface.meta.keypoint import IPointIterator
from meta.keypoint import Point

class DelaunayTrianglation(object):
    """
    delaunay triangulation
    """
    def __init__(self, filepath: Text) -> None:
        points = read_points(filepath)
        self._points = [(pt.x, pt.y) for pt in points]
        self._triangulate()
        self._convex_hull()

    def _triangulate(self) -> None:
        # rect (tuple) - "xmin", "ymin", "w", "h"
        rect = cv2.boundingRect(np.array(self._points))
        subdiv = cv2.Subdiv2D(rect)
        for pt in self._points:
            subdiv.insert(pt)
        self._trianglar_grid = []
        # trangules
        trangules = subdiv.getTriangleList()
        for tri in trangules:
            # vertices of a triangule
            tri = [
                (tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])
            ]
            # point indices of a trangule
            tri_pts = []
            for v in tri:
                # go through all points
                for idx, pt in enumerate(self._points):
                    if (abs(v[0] - pt[0]) < 1) and (abs(v[1] - pt[1]) < 1):
                        tri_pts.append(idx)
            self._trianglar_grid.append(tri_pts)

    def _convex_hull(self) -> None:
        """
        convex hull of points
        """
        self._contour_points = cv2.convexHull(
            points=np.array(self._points), returnPoints=False
        )
        self._contour_points = self._contour_points.flatten().tolist()

    @property
    def trianglar_grid(self):
        return self._trianglar_grid

    @property
    def contour(self):
        return self._contour_points

def read_points(filepath: Text) -> IPointIterator:
    """
    reading points
        x y
        x y
        x y
        ...
        x y
    """
    assert os.path.isfile(filepath), "ERROR: file does not exist!"
    with open(file=filepath, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        pt = Point(*[int(item) for item in line.strip().split(" ")])
        points.append(pt)
    return points

if __name__ == "__main__":
    filepath = "./data/face_swap/hillary_clinton.jpg.txt"
    delaunay = DelaunayTrianglation(filepath)
    print(delaunay.trianglar_grid)
    print(delaunay.contour)
