from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
import torch

from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.datasets.osm import OsmTileDataset


CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-218fbf4b-b33a-40eb-8929-3f006826c0a0_embedding_25000.pth'
gdf = gpd.read_file("data/coastline/coastline.geojson")
AOI = gdf.loc[gdf["adm0_a3"] == "GBR"].geometry.unary_union

embedding_model = ResNetEmbedding(16)
embedding_model.load_state_dict(torch.load(CHECKPOINT))
embedding_model.zero_grad()
embedding_model.eval()
embedding_model.cuda()

dataset = OsmTileDataset(
    AOI,
    sample_count=20_000,
    buffer=100.0,
    distance=250.0,
    seed=2,
    cache_dir=Path("data/cache/embedding_math")
)


class MismatchedExpression(Exception): pass


def load_embeddings() -> List[np.ndarray]:
    print("Initialising embeddings.")
    embeddings = []
    # for anchor, _, _ in (dataset[i] for i in range(100)):
    for anchor, _, _ in (dataset[i] for i in range(len(dataset))):

        if len(embeddings) % 100 == 0:
            print(f"{len(embeddings)} / {len(dataset)}")

        anchor = anchor.unsqueeze(0).cuda()

        embeddings.append(
            embedding_model(anchor).squeeze().detach().cpu().numpy()
        )
    return embeddings


def plot_neighbours(anchor_idx: int, neighbour_idxes: List[int]):
    plt.close('all')

    fig, axes = plt.subplots(len(neighbour_idxes), 2, figsize=(10, 10))
    plt.setp(axes, xticks=[], yticks=[])
    [ax.set_axis_off() for ax in axes.ravel()]

    axes[0][0].imshow(
        dataset.load_triplet_images(anchor_idx)[0]
    )
    axes[0][0].title.set_text(anchor_idx)
    for axes_index, neighbour_idx in enumerate(neighbour_idxes):
        axes[axes_index][1].imshow(
            dataset.load_triplet_images(neighbour_idx)[0]
        )
        axes[axes_index][1].title.set_text(neighbour_idx)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def plot_arithmetic(variable1_idx: int, operator: str, variable2_idx: int, result_idx: int):
    plt.close('all')

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    plt.setp(axes, xticks=[], yticks=[])
    [ax.set_axis_off() for ax in axes.ravel()]
    plt.title(operator)

    axes[0].imshow(
        dataset.load_triplet_images(variable1_idx)[0]
    )
    axes[0].title.set_text(variable1_idx)

    axes[1].imshow(
        dataset.load_triplet_images(variable2_idx)[0]
    )
    axes[1].title.set_text(variable2_idx)

    axes[2].imshow(
        dataset.load_triplet_images(result_idx)[0]
    )
    axes[2].title.set_text(result_idx)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def plot_interpolation(interpolated_idxes: List[int]):
    plt.close('all')

    fig, axes = plt.subplots(1, len(interpolated_idxes), figsize=(10, 5))
    plt.setp(axes, xticks=[], yticks=[])

    for axes_index, idx in enumerate(interpolated_idxes):
        ax = axes[axes_index]
        ax.imshow(
            # OsmTileDataset.load_triplet() returns anchor, positive, negative
            dataset.load_triplet_images(idx)[0]
        )
        ax.title.set_text(idx)

        # Keep the border for the first and last images so that
        # its clear they were selected.
        if axes_index not in (0, len(interpolated_idxes) - 1):
            ax.set_axis_off()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def handle_arithmetic(expression: str, embeddings: List[np.ndarray], nn_model: NearestNeighbors):
    try:
        parts = expression.split(" ")
        variable1_idx = int(parts[0])
        operator = parts[1]
        variable2_idx = int(parts[2])
    except Exception:
        raise MismatchedExpression()

    if operator == "+":
        result_embedding = embeddings[variable1_idx] + embeddings[variable2_idx]
    elif operator == "-":
        result_embedding = embeddings[variable1_idx] - embeddings[variable2_idx]
    else:
        raise MismatchedExpression()

    result_idx = nn_model.kneighbors(
        result_embedding.reshape(1, -1),
        2,
        return_distance=False
    ).tolist()[0][1]

    plot_arithmetic(variable1_idx, operator, variable2_idx, result_idx)
    plt.savefig(f"assets/arithmetic/{expression.replace(' ', '-')}.png")


def handle_interpolation(expression: str, embeddings: List[np.ndarray], nn_model: NearestNeighbors):
    try:
        parts = expression.split(" ")
        idx1 = int(parts[0])
        interpolate = str(parts[1]) == "interp"
        idx2 = int(parts[2])
    except Exception:
        raise MismatchedExpression()

    if not interpolate:
        raise MismatchedExpression()

    steps = 6
    f = interp1d(
        [0, steps],
        np.vstack([embeddings[idx1], embeddings[idx2]]),
        axis=0
    )

    interpolated_idxes = []
    for i in range(0, steps + 1):
        interpolated_idxes.append(
            nn_model.kneighbors(
                f(i).reshape(1, -1),
                1,
                return_distance=False
            ).tolist()[0][0]
        )

    plot_interpolation(interpolated_idxes)
    plt.savefig(f"assets/interpolation/{expression.replace(' ', '-')}.png")


def handle_neighbours(expression: str, embeddings: List[np.ndarray], nn_model: NearestNeighbors):
    try:
        parts = expression.split(" ")
        idx = int(parts[0])
        neighbours = int(parts[1]) + 1
    except Exception:
        raise MismatchedExpression()

    neighbour_idxes = nn_model.kneighbors(
        embeddings[idx].reshape(1, -1),
        neighbours,
        return_distance=False
    ).tolist()[0][1:]

    plot_neighbours(idx, neighbour_idxes)
    plt.savefig(f"assets/nearest_neighbours/{expression.replace(' ', '-')}.png")


def run(embeddings: List[np.ndarray], nn_model: NearestNeighbors):
    expression = input("Enter expression.")

    for handler in [
        handle_arithmetic,
        handle_neighbours,
        handle_interpolation
    ]:
        try:
            handler(expression, embeddings, nn_model)
            return
        except MismatchedExpression:
            # Handler can't parse the expression
            pass
        except Exception as e:
            # Handler thought it could parse the
            # expression but could not
            print(e)

    # No handlers could handle the expression.
    print("Invalid expression")


def main():
    embeddings = load_embeddings()

    model = NearestNeighbors()
    model.fit(embeddings)

    plt.ion()
    plt.show()
    while True:
        run(embeddings, model)


if __name__ == "__main__":
    main()
