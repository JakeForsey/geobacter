import asyncio
from pathlib import Path
import random
from typing import Tuple, List

import httpx
from PIL.Image import Image as PILImage
import skimage.measure
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from geobacter.inference.geotypes import Extent
from geobacter.inference.mapnik import get_extent
from geobacter.inference.datasets.sample import TripletSample

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
            samples: List[TripletSample],
            cache_dir: Path
    ):
        self.samples = samples
        self.negative_extents = [
            sample.anchor.extent for sample in self.samples
        ] + [
            sample.positive.extent for sample in self.samples
        ]
        self.cache_dir = cache_dir
        self.client = httpx.AsyncClient()
        self.index_to_entropy = {}
        print(f"OsmTileDataset initialised. samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

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
        async def _anchor_entropy():
            if index not in self.index_to_entropy:
                self.index_to_entropy[index] = await extent_to_entropy(
                    self.samples[index].anchor.extent,
                    self.cache_dir,
                    self.client
                )
            return self.index_to_entropy[index]

        loop = asyncio.get_event_loop()
        task = loop.create_task(_anchor_entropy())
        return loop.run_until_complete(task)

    def load_triplet_images(self, index: int) -> Tuple[PILImage, PILImage, PILImage]:
        async def _load_triplet():
            a = await get_extent(
                self.samples[index].anchor.extent,
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            self.index_to_entropy[index] = skimage.measure.shannon_entropy(a.convert('LA'))
            p = await get_extent(
                self.samples[index].positive.extent,
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            n = await get_extent(
                random.choice(self.negative_extents),
                cache_dir=self.cache_dir, zoom=16, client=self.client
            )
            return a, p, n

        loop = asyncio.get_event_loop()
        task = loop.create_task(_load_triplet())
        return loop.run_until_complete(task)


async def extent_to_entropy(extent: Extent, cache_dir: Path, client: httpx.AsyncClient) -> float:
    image = await get_extent(
        extent,
        cache_dir=cache_dir, zoom=16, client=client
    )
    return skimage.measure.shannon_entropy(image.convert('LA'))
