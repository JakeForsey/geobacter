import argparse
from pathlib import Path
import pickle

import geopandas as gpd

from geobacter.inference.datasets.osm import generate_extents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--buffer", type=float)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--seed", type=float)
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()

    gdf = gpd.read_file("data/coastline/coastline.geojson")
    aoi = gdf.loc[gdf["adm0_a3"] == "GBR"].geometry.unary_union

    anchor_extents, positive_extents = generate_extents(
        aoi,
        args.sample_count,
        args.buffer,
        args.distance,
        args.seed
    )

    with args.path.open("wb") as f:
        pickle.dump((anchor_extents, positive_extents), f)
