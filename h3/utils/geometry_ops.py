from __future__ import annotations

from shapely.geometry import Polygon, Point


def center_poly(poly: Polygon) -> Point:
	return poly.centroid
