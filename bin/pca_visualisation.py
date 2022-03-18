from pathlib import Path
from typing import List
import random

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import trange

from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.datasets.osm import OsmTileDataset
from geobacter.inference.datasets.sample import load_samples

CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-d5abe7e5-3ac9-4b24-8e27-cd8d432a7ec9_embedding_1105.pth'
CACHE_DIR = Path("data/cache")

embedding_model = ResNetEmbedding(16)
embedding_model.load_state_dict(torch.load(CHECKPOINT))
embedding_model.zero_grad()
embedding_model.eval()
embedding_model.cuda()

dataset = OsmTileDataset(
    samples=[sample for sample in load_samples(Path("data/extents/test_10000.json"))
             if random.random() > 0.99 or sample.anchor.entropy > 1.7],
    cache_dir=CACHE_DIR
)


def load_embeddings() -> List[np.ndarray]:
    print("Initialising embeddings.")
    embeddings = []
    for anchor, _, _ in (dataset[i] for i in trange(len(dataset), desc="loading embeddings")):
        anchor = anchor.unsqueeze(0).cuda()
        embeddings.append(
            embedding_model(anchor).squeeze().detach().cpu().numpy()
        )
    return embeddings


def main():
    embeddings = load_embeddings()

    rgb = PCA(n_components=3).fit_transform(embeddings)
    rgb = MinMaxScaler((0, 255)).fit_transform(rgb)
    extents = [dataset.sample(i).anchor.extent for i in trange(len(dataset), desc="loading metadata")]

    gdf = gpd.GeoDataFrame(
        {
         "colour": [",".join([str(int(round(v))) for v in rgb[i].tolist()]) for i in range(len(extents))]
        },
        geometry=[box(*e) for e in extents]
    )
    print(gdf.head())
    gdf.to_file("rgb.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
