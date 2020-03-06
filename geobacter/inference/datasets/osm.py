import asyncio
from pathlib import Path
import pickle
import random
from typing import Tuple, List

import httpx
from PIL.Image import Image as PILImage
import skimage.measure
from shapely.geometry import Polygon
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from tqdm import tqdm

from geobacter.inference.util import random_point
from geobacter.inference.util import random_translation
from geobacter.inference.util import buffer_point
from geobacter.inference.geotypes import Meters, Extent
from geobacter.inference.mapnik import get_extent

AUGMENTATIONS = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip()
])
BASE_TRANSFORMS = Compose([
    Resize((128, 128)),
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
            extents_path: Path,
            cache_dir: Path
    ):
        with extents_path.open("rb") as f:
            anchor_extents, positive_extents = pickle.load(f)

        self.anchor_extents = []
        self.positive_extents = []
        self.cache_dir = cache_dir
        self.client = httpx.AsyncClient()
        self.index_to_entropy = {}

        for a_extent, p_extent in tqdm(zip(anchor_extents, positive_extents), total=len(anchor_extents)):
            if random.random() > 0.99 or extent_to_entropy(a_extent, self.cache_dir, self.client) > 1.8:
                self.anchor_extents.append(a_extent)
                self.positive_extents.append(p_extent)

        print(len(self.anchor_extents))

    def __len__(self):
        return len(self.anchor_extents)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, positive, negative = self.load_triplet_images(index)

        anchor = BASE_TRANSFORMS(
            AUGMENTATIONS(
                anchor
            )
        )
        positive = BASE_TRANSFORMS(
            AUGMENTATIONS(
                positive
            )
        )
        negative = BASE_TRANSFORMS(
            AUGMENTATIONS(
                negative
            )
        )

        return anchor, positive, negative

    def anchor_entropy(self, index: int) -> float:
        if index not in self.index_to_entropy:
            self.index_to_entropy[index] = extent_to_entropy(
                self.anchor_extents[index],
                self.cache_dir,
                self.client
            )

        return self.index_to_entropy[index]

    def load_triplet_images(self, index: int) -> Tuple[PILImage, PILImage, PILImage]:
        async def _load_triplet():
            a = await get_extent(
                self.anchor_extents[index],
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            self.index_to_entropy[index] = skimage.measure.shannon_entropy(a.convert('LA'))
            p = await get_extent(
                self.positive_extents[index],
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            n = await get_extent(
                random.choice(self.anchor_extents + self.positive_extents),
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            return a, p, n

        loop = asyncio.get_event_loop()
        task = loop.create_task(_load_triplet())
        return loop.run_until_complete(task)


def extent_to_entropy(extent: Extent, cache_dir: Path, client: httpx.AsyncClient) -> float:
    async def _load_anchor():
        return await get_extent(
            extent,
            cache_dir=cache_dir, zoom=16, client=client
        )

    loop = asyncio.get_event_loop()
    task = loop.create_task(_load_anchor())
    anchor = loop.run_until_complete(task)
    return skimage.measure.shannon_entropy(anchor.convert('LA'))


def generate_extents(
        aoi: Polygon,
        sample_count: int,
        buffer: Meters,
        distance: Meters,
        seed: int,
) -> Tuple[List[Extent], List[Extent]]:
    random.seed(seed)
    anchor_points = [random_point(aoi) for _ in range(sample_count)]
    positive_points = [random_translation(point, distance) for point in anchor_points]

    anchor_extents = [buffer_point(point, buffer) for point in anchor_points]
    positive_extents = [buffer_point(point, buffer) for point in positive_points]

    return anchor_extents, positive_extents
