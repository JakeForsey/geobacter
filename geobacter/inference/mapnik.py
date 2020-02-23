from io import BytesIO
from pathlib import Path
from PIL import Image

import httpx
from geobacter.inference.constants import TILE_WIDTH, TILE_HEIGHT
from geobacter.inference.geotypes import Extent, Point
from geobacter.inference.util import point_to_slippy_tile
from geobacter.inference.util import slippy_tile_to_point


async def load_extent(
    extent: Extent,
    cache_dir: Path,
    zoom: int,
) -> 'Image':
    path = extent_to_path(extent, cache_dir, zoom)
    if path.is_file():
        image = Image.open(path)
    else:
        image = cache_extent(extent, cache_dir, zoom)

    return image


async def cache_extent(
    extent: Extent,
    cache_dir: Path,
    zoom: int
):
    path = extent_to_path(extent, cache_dir, zoom)
    image = await request_extent(extent, zoom)
    image.save(path)
    return image


def extent_to_path(
        extent: Extent,
        cache_dir: Path,
        zoom: int,
) -> Path:
    return cache_dir / Path(f"{zoom}/{extent}.png")


async def request_extent(extent: Extent, zoom: int) -> 'Image':
    x_min, y_min = point_to_slippy_tile((extent[0], extent[1]), zoom)
    x_max, y_max = point_to_slippy_tile((extent[2], extent[3]), zoom)
    print(x_min, y_min)
    print(x_max, y_max)

    mosaic = Image.new(
        'RGB',
        # TODO Think about these -1s
        ((x_max - x_min + 1) * TILE_WIDTH, (y_max - y_min + 1) * TILE_HEIGHT)
    )
    tiles = {}
    for x in range(x_min, x_max + 1):
        for y in range(y_min,  y_max + 1):
            tiles[(x, y)] = await request_tile(x, y, zoom)

    for (x, y), tile in tiles.items():
        print(x, y)
        mosaic.paste(
            tile,
            box=(
                (x - x_min) * TILE_WIDTH,
                (y - y_min) * TILE_HEIGHT
            )
        )

    mosaic.save("mosaic.png")
    extent_left, extent_upper, extent_right, extent_lower = extent
    extent_width = extent_right - extent_left

    mosaic_left, mosaic_upper = slippy_tile_to_point(x_min, y_min, zoom)
    mosaic_right, mosaic_lower = slippy_tile_to_point(x_max, y_max, zoom)

    mosaic_width, mosaic_height = mosaic.size

    image = mosaic.crop(
        box=(
            # left, upper, right, lower
            (mosaic_left - extent_left) * mosaic_width,
            (mosaic_upper - extent_upper) * mosaic_height,
            (mosaic_left - extent_left) * mosaic_width,
            (mosaic_left - extent_left) * mosaic_width,
        )
    )
    image.save("image.png")
    return image


async def request_tile(x: int, y: int, zoom: int) -> 'Image':
    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8080/tile/{zoom}/{x}/{y}.png"
        print(url)
        while True:
            try:
                response = await client.get(url, timeout=None)
            except httpx.exceptions.ConnectionClosed:
                print("Connection closed, retrying.")
            if response.status_code == 200:
                break

    return Image.open(BytesIO(response.content))


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        request_extent(
            (-2.8, 54.0, -0.2, 51.0),
            4
        )
    )
    anchor, positive, negative = loop.run_until_complete(task)


