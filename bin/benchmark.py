from abc import ABC
from abc import abstractmethod
import asyncio
from itertools import islice
from functools import lru_cache
import math
from pathlib import Path
import random
from typing import Iterable

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm

from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.mapnik import get_extent
from geobacter.inference.geotypes import Point
from geobacter.inference.util import buffer_point
from geobacter.inference.datasets.osm import BASE_TRANSFORMS


CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-c448224c-a38e-4c02-8b8c-572ff00e21db_embedding_45297.pth'
CACHE_DIR = Path("data/cache")
TRAIN_SAMPLES = [10, 100, 1_000, 10_000]
embedding_model = ResNetEmbedding(16)
embedding_model.load_state_dict(torch.load(CHECKPOINT))
embedding_model.zero_grad()
embedding_model.eval()
embedding_model.cuda()


class Benchmark(ABC):
    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def problem_type(self):
        pass

    @abstractmethod
    def _points(self):
        pass

    @property
    @lru_cache()
    def embeddings(self) -> np.ndarray:
        async def point_to_embedding(point):
            async with httpx.AsyncClient() as client:
                image = await get_extent(
                    buffer_point(point, 100.0),
                    CACHE_DIR,
                    16,
                    client
                )
                image = BASE_TRANSFORMS(image)
                image = image.cuda()
                return embedding_model(image.unsqueeze(0)).detach().cpu().numpy()

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        loop = asyncio.get_event_loop()
        chunk_size = 10
        embeddings = []
        with tqdm(total=len(self._points())) as pbar:
            for points in chunk(self._points(), chunk_size):
                coroutines = [point_to_embedding(point) for point in points]
                embeddings.extend(
                    loop.run_until_complete(
                        asyncio.gather(*coroutines)
                    )
                )

                pbar.update(chunk_size)

        return np.concatenate(embeddings)


class OutputAreaClassification(Benchmark):
    def __init__(self):
        self.df = pd.read_csv("data/benchmarks/output_area_classification/data.csv").sample(20_000, random_state=1)

    @property
    def X(self):
        X = self.df[[
            "Region Name",
            "Longitude",
            "Latitude",
        ]].copy()
        numeric_dtypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_columns = []
        for col in X.columns.values.tolist():
            if X[col].dtype not in numeric_dtypes:
                categorical_columns.append(col)
                X[col] = X[col].astype(str).fillna("NA")
            else:
                X[col] = X[col].fillna(X[col].mean())

        for col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col].values)

        return X.copy()

    @property
    def y(self):
        return LabelEncoder().fit_transform(self.df["Output Area Classification Name"].values)

    @property
    def problem_type(self):
        return "classification"

    def _points(self) -> Iterable[Point]:
        return [(row["Longitude"], row["Latitude"]) for idx, row in self.df.iterrows()]


def evaluate(benchmark: Benchmark, train_size: int, seed=1):

    def _score(X_train, X_test, y_train, y_test):
        m = RandomForestRegressor() if benchmark.problem_type == "regression" else RandomForestClassifier()
        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)
        from functools import partial
        score_fn = r2_score if benchmark.problem_type == "regression" else partial(f1_score, average="macro")
        return score_fn(y_test, y_pred)

    X_train, X_test, y_train, y_test, embeddings_train, embeddings_test = train_test_split(
        benchmark.X, benchmark.y, benchmark.embeddings,
        train_size=train_size,
        random_state=seed
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    baseline_score = _score(X_train, X_test, y_train, y_test)
    embeddings_score = _score(embeddings_train, embeddings_test, y_train, y_test)

    combined_train = np.concatenate([embeddings_train, X_train], axis=1)
    combined_test = np.concatenate([embeddings_test, X_test], axis=1)
    combined_score = _score(combined_train, combined_test, y_train, y_test)

    return {
        "train_size": train_size,
        "baseline_score": baseline_score,
        "embeddings_score": embeddings_score,
        "combined_score": combined_score,
    }


BENCHMARKS = {
    "OutputAreaClassification": OutputAreaClassification,
}


def benchmark_from_name(name: str):
    return BENCHMARKS[name]()


def run_all():
    random.seed()

    for name in BENCHMARKS:
        benchmark = benchmark_from_name(name)

        results = []
        for seed in [1, 2, 3]:
            for train_size in TRAIN_SAMPLES:
                print(f"Evaluating benchmark with {train_size} training examples (seed={seed}).")
                results.append(evaluate(benchmark, train_size, seed))

        print(results)
        df = pd.DataFrame(results)

        plt.scatter(
            df.train_size.apply(lambda x: math.log(x, 10)),
            df.baseline_score,
            c="magenta",
            marker="D",
            alpha=0.5,
            label="features only",
        )
        plt.scatter(
            df.train_size.apply(lambda x: math.log(x, 10)),
            df.embeddings_score,
            c="cyan",
            marker="s",
            alpha=0.5,
            label="embeddings only",
        )
        plt.scatter(
            df.train_size.apply(lambda x: math.log(x, 10)),
            df.combined_score,
            c="blue",
            marker="^",
            alpha=0.5,
            label="embeddings and features",
        )

        summary = df.groupby("train_size").mean().reset_index()
        plt.plot(
            summary.train_size.apply(lambda x: math.log(x, 10)),
            summary.baseline_score,
            "--",
            c="magenta",
            label="features only"
        )
        plt.plot(
            summary.train_size.apply(lambda x: math.log(x, 10)),
            summary.embeddings_score,
            "--",
            c="cyan",
            label="embeddings only",
        )
        plt.plot(
            summary.train_size.apply(lambda x: math.log(x, 10)),
            summary.combined_score,
            "--",
            c="blue",
            label="embeddings and features",
        )
        plt.plot()
        plt.xlabel("log₁₀(train samples)")
        plt.ylabel("R²", rotation=0)
        plt.xticks([math.log(i, 10) for i in TRAIN_SAMPLES])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{benchmark.__class__.__name__}.png")
        plt.clf()


if __name__ == "__main__":
    run_all()
