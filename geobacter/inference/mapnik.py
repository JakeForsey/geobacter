from io import BytesIO
from pathlib import Path
from PIL import Image

import httpx

from geobacter.inference.constants import TILE_WIDTH, TILE_HEIGHT
from geobacter.inference.geotypes import Extent
from geobacter.inference.util import point_to_slippy_tile
from geobacter.inference.util import slippy_tile_to_point


async def get_extent(
        extent: Extent,
        cache_dir: Path,
        zoom: int,
        client: httpx.AsyncClient
) -> 'Image':
    path = extent_to_path(extent, cache_dir, zoom)
    if path.is_file():
        image = Image.open(path)
    else:
        image = await cache_extent(extent, cache_dir, zoom, client)

    return image


async def cache_extent(
        extent: Extent,
        cache_dir: Path,
        zoom: int,
        client: httpx.AsyncClient
):
    path = extent_to_path(extent, cache_dir, zoom)
    image = await request_extent(extent, zoom, client)
    try:
        image.save(path)
    except SystemError:
        path.unlink()
        raise RuntimeWarning("Failed to save image to cache.")
    except FileNotFoundError:
        path.parent.mkdir(exist_ok=True, parents=True)
        image.save(path)

    return image


def extent_to_path(
        extent: Extent,
        cache_dir: Path,
        zoom: int,
) -> Path:
    return cache_dir / Path(f"{zoom}/{','.join([str(v) for v in extent])}.png")


async def request_extent(extent: Extent, zoom: int, client: httpx.AsyncClient) -> 'Image':
    x_min, y_max = point_to_slippy_tile((extent[0], extent[1]), zoom)
    x_max, y_min = point_to_slippy_tile((extent[2], extent[3]), zoom)
    x_max += 1
    y_max += 1

    mosaic = Image.new(
        'RGB',
        ((x_max - x_min) * TILE_WIDTH, (y_max - y_min) * TILE_HEIGHT)
    )
    tiles = {}
    for x in range(x_min, x_max):
        for y in range(y_min,  y_max):
            tiles[(x, y)] = await request_tile(x, y, zoom, client)

    for (x, y), tile in tiles.items():
        mosaic.paste(
            tile,
            box=(
                (x - x_min) * TILE_WIDTH,
                (y - y_min) * TILE_HEIGHT
            )
        )

    extent_left, extent_upper, extent_right, extent_lower = extent

    mosaic_left, mosaic_upper = slippy_tile_to_point(x_min, y_min, zoom)
    mosaic_right, mosaic_lower = slippy_tile_to_point(x_max, y_max, zoom)

    mosaic_width, mosaic_height = mosaic_right - mosaic_left, mosaic_upper - mosaic_lower
    mosaic_width_px, mosaic_height_px = mosaic.size

    box = (
        # left, upper, right, lower
        int(round((abs(mosaic_left - extent_left) / mosaic_width) * mosaic_width_px)),
        int(round((abs(mosaic_upper - extent_lower) / mosaic_height) * mosaic_height_px)),
        int(round((abs(mosaic_left - extent_right) / mosaic_width) * mosaic_width_px)),
        int(round((abs(mosaic_upper - extent_upper) / mosaic_height) * mosaic_height_px)),
    )
    image = mosaic.crop(
        box=box
    )
    return image


async def request_tile(x: int, y: int, zoom: int, client: httpx.AsyncClient) -> 'Image':
    url = f"http://localhost:8080/tile/{zoom}/{x}/{y}.png"
    while True:
        try:
            response = await client.get(url, timeout=None)
            if response.status_code == 200:
                break
        except httpx.exceptions.ConnectionClosed:
            print("Connection closed, retrying.")

    return Image.open(BytesIO(response.content))
