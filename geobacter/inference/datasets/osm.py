import asyncio
from PIL import Image
import random
from typing import Callable, Tuple, Awaitable

from shapely.geometry import Polygon
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from geobacter.inference.util import random_point
from geobacter.inference.util import random_translation
from geobacter.inference.util import buffer_point
from geobacter.inference.geotypes import Extent

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
            aoi: Polygon,
            sample_count: int,
            buffer: float,
            load_extent_fn: Callable[[Extent], Awaitable['Image']]
    ):
        anchor_points = [random_point(aoi, seed=1) for _ in range(sample_count)]
        positive_points = [random_translation(point, seed=1) for point in anchor_points]

        self.sample_count = sample_count
        self.anchor_extents = [buffer_point(point, buffer) for point in anchor_points]
        self.positive_extents = [buffer_point(point, buffer) for point in positive_points]
        self.load_extent = load_extent_fn

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int) -> Tuple['Image', 'Image', 'Image']:
        async def load_triplet():
            return (
                await self.load_extent(self.anchor_extents[index]),
                await self.load_extent(self.positive_extents[index]),
                await self.load_extent(random.choice(self.anchor_extents + self.positive_extents))
            )

        loop = asyncio.get_event_loop()
        task = loop.create_task(load_triplet())
        anchor, positive, negative = loop.run_until_complete(task)

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
