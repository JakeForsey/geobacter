from dataclasses import dataclass
from pathlib import Path
import json
from typing import Tuple, List, Dict

from geobacter.inference.geotypes import Extent, Point


@dataclass
class Location:
    point: Point
    extent: Extent
    entropy: float


@dataclass
class TripletSample:
    anchor: Location
    positive: Location


def load_samples(path: Path) -> List[TripletSample]:
    print(f"Loading samples from {path}.")

    with path.open() as f:
        data = json.load(f)
        samples = data.pop("samples")
        print(f"Sample metadata: {data}")
        samples = [TripletSample(
            Location(
                sample["anchor"]["point"],
                sample["anchor"]["extent"],
                sample["anchor"]["shannon_entropy"]
            ),
            Location(
                sample["positive"]["point"],
                sample["positive"]["extent"],
                sample["positive"]["shannon_entropy"]
            )
        ) for sample in samples]

    return samples
