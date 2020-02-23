import math
import random
from typing import Tuple

from shapely.geometry import Polygon

from geobacter.inference.geotypes import Extent, Point


def point_to_slippy_tile(point: Point, zoom: int) -> Tuple[int, int]:
    """
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
    """
    lon_deg, lat_deg = point
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def slippy_tile_to_point(xtile: int, ytile: int, zoom: int) -> Point:
    """
    Returns the NW-corner of the tile.
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def buffer_point(point: Point, buffer: float) -> Extent:
    return point[0] - buffer, point[1] - buffer, point[0] + buffer, point[1] + buffer


def random_point(aoi: Polygon, seed: int = 1) -> Point:
    random.seed(seed)
    minx, miny, maxx, maxy = aoi.bounds
    while True:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if aoi.contains(point):
            return point


def random_translation(point: Point, seed: int = 1) -> Point:
    random.seed(seed)
    return point
