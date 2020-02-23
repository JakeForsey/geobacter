from abc import ABC
from abc import abstractmethod
import asyncio
from itertools import islice
from functools import lru_cache
import math
from pathlib import Path
from PIL import Image
import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import torch

from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.mapnik import get_extent
from geobacter.inference.geotypes import Point
from geobacter.inference.util import buffer_point
from geobacter.inference.datasets.osm import BASE_TRANSFORMS


CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-9e743660-309c-43fd-b50b-9efbc9bdde9d_embedding_100000.pth'
HOUSE_PRICES = "data/benchmarks/house_prices/data.csv"
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

    @abstractmethod
    def _points(self):
        pass

    @abstractmethod
    def _cache_directory(self) -> Path:
        pass

    @property
    @lru_cache()
    def embeddings(self) -> np.ndarray:
        self._cache_directory().mkdir(exist_ok=True, parents=True)

        async def point_to_embedding(point):
            image = get_extent(
                buffer_point(point, 100.0),
                self._cache_directory(),
                17
            )
            image = BASE_TRANSFORMS(image)
            image = image.cuda()
            return embedding_model(image.unsqueeze(0)).detach().cpu().numpy()

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        loop = asyncio.get_event_loop()
        embeddings = []
        for points in chunk(self._points(), 10):
            coroutines = [point_to_embedding(point) for point in points]
            embeddings.extend(
                loop.run_until_complete(
                    asyncio.gather(*coroutines)
                )
            )

        return np.concatenate(embeddings)


class HousePriceBenchmark(Benchmark):
    def __init__(self):
        self.df = pd.read_csv(Path("data/benchmarks/house_prices/data.csv"))

    @property
    def X(self):
        return self.df[["sqft_living", "bedrooms", "bathrooms"]]

    @property
    def y(self):
        return self.df["price"]

    def _points(self) -> Iterable[Point]:
        return [(row["long"], row["lat"]) for idx, row in self.df.iterrows()]

    def _cache_directory(self) -> Path:
        return Path("data/cache/benchmarks/house_prices")


class CarPriceBenchmark(Benchmark):
    def __init__(self):
        df = pd.read_csv(Path("data/benchmarks/car_prices/data.csv"))

        self.df = df[
            df["lat"].between(46.124742, 48.880009) &
            df["long"].between(-124.472116, -117.176261)
        ].copy()

    @property
    def X(self):
        X = self.df[[
            "manufacturer",
            "year",
            "odometer",
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

        X['year'] = (X['year'] - 1900).astype(int)
        X['odometer'] = X['odometer'].astype(int)

        return X.copy()

    @property
    def y(self):
        return self.df["price"]

    def _points(self) -> Iterable[Point]:
        return [(row["long"], row["lat"]) for idx, row in self.df.iterrows()]

    def _cache_directory(self) -> Path:
        return Path("data/cache/benchmarks/car_prices")


def evaluate(benchmark: Benchmark, train_size: int, seed=1):

    def _score(X_train, X_test, y_train, y_test):
        m = RandomForestRegressor()
        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)
        return r2_score(y_test, y_pred)

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
    "HousePrice": HousePriceBenchmark,
    "CarPrice": CarPriceBenchmark
}


def benchmark_from_name(name: str):
    return BENCHMARKS[name]()


def run():
    random.seed()

    for name in BENCHMARKS:
        benchmark = benchmark_from_name(name)

        # results = [{'train_size': 10, 'baseline_score': 205161936.92804074, 'embeddings_score': 214600590.65082082, 'combined_score': 210226430.32083264}, {'train_size': 100, 'baseline_score': 165026683.71304765, 'embeddings_score': 195860592.89620247, 'combined_score': 149466121.35509643}, {'train_size': 1000, 'baseline_score': 127832818.05935825, 'embeddings_score': 154662908.88608417, 'combined_score': 118873809.24728106}, {'train_size': 10000, 'baseline_score': 88796490.04219109, 'embeddings_score': 138480576.19776294, 'combined_score': 83991446.87797993}, {'train_size': 10, 'baseline_score': 168334755.54410255, 'embeddings_score': 234727750.14422667, 'combined_score': 203676938.43009853}, {'train_size': 100, 'baseline_score': 163084136.47468862, 'embeddings_score': 193718346.9897234, 'combined_score': 149031993.87075284}, {'train_size': 1000, 'baseline_score': 127349466.18659735, 'embeddings_score': 152614545.04581943, 'combined_score': 120527263.51661347}, {'train_size': 10000, 'baseline_score': 80592420.19248538, 'embeddings_score': 136406282.38250282, 'combined_score': 82427695.7411659}, {'train_size': 10, 'baseline_score': 234150692.57971254, 'embeddings_score': 261371957.14583448, 'combined_score': 202052559.2870032}, {'train_size': 100, 'baseline_score': 171358586.88289893, 'embeddings_score': 189243108.2399285, 'combined_score': 151590432.94680554}, {'train_size': 1000, 'baseline_score': 159044879.2130711, 'embeddings_score': 160645831.08412796, 'combined_score': 118286496.13325292}, {'train_size': 10000, 'baseline_score': 88006314.70122786, 'embeddings_score': 138165216.84902355, 'combined_score': 85791009.52635176}]
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
    run()
