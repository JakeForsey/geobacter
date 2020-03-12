import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import json
import random
from uuid import uuid4

import httpx
import geopandas as gpd
from tqdm import trange

from geobacter.inference.datasets.osm import extent_to_entropy
from geobacter.inference.util import random_point, random_translation, buffer_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--buffer", type=float)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--seed", type=float)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))

    args = parser.parse_args()

    gdf = gpd.read_file("data/coastline/coastline.geojson")
    aoi = gdf.loc[gdf["adm0_a3"] == "GBR"].geometry.unary_union

    semaphore = asyncio.BoundedSemaphore(10)

    async def generate_sample(client: httpx.AsyncClient):
        async with semaphore:
            anchor_point = random_point(aoi)
            anchor_extent = buffer_point(anchor_point, args.buffer)

            positive_point = random_translation(anchor_point, args.distance)
            positive_extent = buffer_point(positive_point, args.buffer)

            sample = {
                "anchor": {
                    "point": anchor_point,
                    "extent": anchor_extent,
                    "shannon_entropy": await extent_to_entropy(anchor_extent, args.cache_dir, client)
                },
                "positive": {
                    "point": positive_point,
                    "extent": positive_extent,
                    "shannon_entropy": await extent_to_entropy(positive_extent, args.cache_dir, client)
                },
                "sample_uuid": str(uuid4())
            }
            return sample

    async def generate_samples():
        random.seed(args.seed)
        async with httpx.AsyncClient() as client:
            return [await generate_sample(client) for _ in trange(args.sample_count)]

    loop = asyncio.get_event_loop()
    task = loop.create_task(generate_samples())
    samples = loop.run_until_complete(task)
    loop.close()

    with args.path.open("w") as f:
        json.dump(
            {
                "samples": samples,
                "datetime": datetime.now().isoformat(),
                "distance": args.distance,
                "buffer": args.buffer,
                "seed": args.seed
            },
            f
        )
