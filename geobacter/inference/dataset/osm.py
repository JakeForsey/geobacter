import asyncio
import httpx
from io import BytesIO
import math
from pathlib import Path
from PIL import Image
import random
from typing import Tuple
from uuid import uuid4

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

Extent = Tuple[float, float, float, float]
Point = Tuple[float, float]

TILE_HEIGHT = 256
TILE_WIDTH = 256
EXTENTS = {
    "ENGLAND": (-2.8, 54.0, -0.2, 51.0),
    "MIDLANDS": (-1.5, 53.0, -1.0, 53.5)
}

AUGMENTATIONS = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip()
])
BASE_TRANSFORMS = Compose([
    RandomCrop(128 + 64),
    Resize(128),
    ToTensor(),
    Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])
# Reverse the mean std normalization (useful for visualising
# images in Tensorboard.
DENORMALIZE = Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    [1 / 0.229, 1 / 0.224, 1 / 0.225]
)


class OsmTileDataset(Dataset):

    def __init__(
            self,
            cache_dir: Path,
            sample_count: int,
            extent: Extent = EXTENTS["ENGLAND"],
            zoom: int = 16
    ):
        self.cache_dir = cache_dir
        self.sample_count = sample_count
        self.extent = extent
        self.zoom = zoom

        self.semaphore = asyncio.Semaphore(20)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_paths = list(cache_dir.glob("*.png"))

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int) -> Tuple['Image', 'Image', 'Image']:
        triplet = self.load_triplet(index)
        anchor_image, positive_image, negative_image = split(triplet)

        anchor = BASE_TRANSFORMS(
            AUGMENTATIONS(
                anchor_image
            )
        )
        positive = BASE_TRANSFORMS(
            AUGMENTATIONS(
                positive_image
            )
        )
        negative = BASE_TRANSFORMS(
            AUGMENTATIONS(
                negative_image
            )
        )
        return anchor, positive, negative

    def unique_colours(self, index: int) -> int:
        triplet = self.load_triplet(index)
        anchor, _, _ = split(triplet)
        return len(list(filter(lambda x: x[0] > 50, anchor.getcolors())))

    def load_triplet(self, index: int) -> 'Image':
        try:
            image = Image.open(self.image_paths[index])
        except IndexError:
            self.image_paths = list(self.cache_dir.glob("*.png"))
            loop = asyncio.get_event_loop()
            task = loop.create_task(
                cache_triplet(self.cache_dir, self.extent, self.zoom, self.semaphore)
            )
            path, image = loop.run_until_complete(task)
            self.image_paths.append(path)

        return image


def point_to_tile(point: Point, zoom: int) -> Tuple[int, int]:
    """
    Returns the slippy tile index for a point.
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
    """
    lon_deg, lat_deg = point
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_point(xtile: int, ytile: int, zoom: int) -> Point:
    """
    Returns the NW-corner of the tile.
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


async def cache_triplet(
        cache_dir: Path,
        extent: Extent,
        zoom: int,
        semaphore: asyncio.Semaphore
) -> Tuple[Path, 'Image']:
    anchor_point = random_point(extent)
    anchor_x, anchor_y = point_to_tile(anchor_point, zoom)
    anchor_tile = await get_tile(anchor_x, anchor_y, zoom, semaphore)

    offsets = [1, 2, -1, -2]
    positive_x, positive_y = anchor_x + random.choice(offsets), anchor_y + random.choice(offsets)
    positive_tile = await get_tile(positive_x, positive_y, zoom, semaphore)

    negative_point = random_point(extent)
    negative_x, negative_y = point_to_tile(negative_point, zoom)
    negative_tile = await get_tile(negative_x, negative_y, zoom, semaphore)

    image = combine(anchor_tile, positive_tile, negative_tile)
    path = cache_dir / Path(key(anchor_point, zoom))
    image.save(path)
    return path, image


async def cache_tile(
        cache_dir: Path,
        point: Point,
        zoom: int,
        semaphore: asyncio.Semaphore
) -> Tuple[Path, 'Image']:
    x, y = point_to_tile(point, zoom)
    tile = await get_tile(x, y, zoom, semaphore)
    path = cache_dir / Path(key(point, zoom))
    tile.save(path)
    return path, tile


def combine(anchor: 'Image', positive: 'Image', negative: 'Image') -> 'Image':
    image = Image.new("RGB", (3 * TILE_WIDTH, TILE_HEIGHT))
    for idx, tile in enumerate([anchor, positive, negative]):
        image.paste(tile, (idx * TILE_WIDTH, 0))

    return image


def split(image: 'Image') -> Tuple['Image', 'Image', 'Image']:
    return (
        image.crop((0 * TILE_WIDTH, 0, 1 * TILE_WIDTH, TILE_HEIGHT)),
        image.crop((1 * TILE_WIDTH, 0, 2 * TILE_WIDTH, TILE_HEIGHT)),
        image.crop((2 * TILE_WIDTH, 0, 3 * TILE_WIDTH, TILE_HEIGHT)),
    )


def key(point: Point, zoom: int) -> Path:
    x, y = point_to_tile(point, zoom)
    return Path(f"{zoom}-{x}-{y}.png")


def point_to_extent(point: Point, buffer: float) -> Extent:
    return point[0] - buffer, point[1] - buffer, point[0] + buffer, point[1] + buffer


def random_point(extent: Extent) -> Point:
    return random.uniform(extent[0], extent[2]), random.uniform(extent[1], extent[3])


async def get_tile_from_point(point: Point, zoom: int, semaphore: asyncio.Semaphore) -> 'Image':
    x, y = point_to_tile(point, zoom)
    return await get_tile(x, y, zoom, semaphore)


async def get_tile(x: int, y: int, zoom: int, semaphore: asyncio.Semaphore) -> 'Image':

    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8080/tile/{zoom}/{x}/{y}.png"

        while True:
            async with semaphore:
                try:
                    response = await client.get(url, timeout=None)
                except httpx.exceptions.ConnectionClosed:
                    print("Connection closed, retrying.")
            if response.status_code == 200:
                break

    return Image.open(BytesIO(response.content))
