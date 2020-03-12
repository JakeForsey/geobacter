import math
import random
from typing import Tuple

from shapely.geometry import Polygon
from shapely.geometry import Point as ShapelyPoint
import utm

from geobacter.inference.geotypes import Extent, Meters, Point


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
    return lon_deg, lat_deg


def buffer_point(point: Point, buffer: Meters) -> Extent:
    x, y, zone_number, zone_letter = utm.from_latlon(point[1], point[0])
    left = x - buffer
    upper = y - buffer
    right = x + buffer
    lower = y + buffer

    upper, left = utm.to_latlon(left, upper, zone_number, zone_letter)
    lower, right = utm.to_latlon(right, lower, zone_number, zone_letter)

    return left, upper, right, lower


def random_point(aoi: Polygon) -> Point:
    minx, miny, maxx, maxy = aoi.bounds
    while True:
        point = ShapelyPoint(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if aoi.contains(point):
            return point.x, point.y


def random_translation(point: Point, distance: Meters) -> Point:
    x, y, zone_number, zone_letter = utm.from_latlon(point[1], point[0])

    angle = math.radians(random.randrange(0, 360))
    x += distance * math.sin(angle)
    y += distance * math.cos(angle)

    lat, lon = utm.to_latlon(x, y, zone_number, zone_letter)
    return lon, lat
