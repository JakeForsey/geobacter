from abc import ABC
from abc import abstractmethod
import asyncio
from pathlib import Path
import random
from typing import Iterable
from PIL import Image

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch

from geobacter.inference.model.triplet import ResNetEmbedding
from geobacter.inference.dataset.osm import Point
from geobacter.inference.dataset.osm import cache_tile
from geobacter.inference.dataset.osm import BASE_TRANSFORMS
from geobacter.inference.dataset.osm import key


CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-9e743660-309c-43fd-b50b-9efbc9bdde9d_embedding_100000.pth'
HOUSE_PRICES = "data/benchmarks/house_prices/data.csv"
embedding_model = ResNetEmbedding(16)
embedding_model.load_state_dict(torch.load(CHECKPOINT))
embedding_model.zero_grad()
embedding_model.eval()
embedding_model.cuda()


class Benchmark(ABC):
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @abstractmethod
    def y(self) -> np.ndarray:
        pass

    @abstractmethod
    def _points(self):
        pass

    @abstractmethod
    def _cache_directory(self) -> Path:
        pass

    def embeddings(self) -> np.ndarray:
        self._cache_directory().mkdir(exist_ok=True, parents=True)

        semaphore = asyncio.Semaphore(20)

        async def point_to_embedding(point):
            file_path = self._cache_directory() / key(point, 16)

            if not file_path.is_file():
                _, tile = await cache_tile(
                    self._cache_directory(),
                    point,
                    16,
                    semaphore
                )
            else:
                tile = Image.open(file_path)

            image = BASE_TRANSFORMS(tile)
            image = image.cuda()
            return embedding_model(image.unsqueeze(0)).detach().cpu().numpy()

        loop = asyncio.get_event_loop()
        coroutines = [point_to_embedding(point) for point in self._points()]
        embeddings = loop.run_until_complete(
            asyncio.gather(*coroutines)
        )

        return np.concatenate(embeddings)


class HousePriceBenchmark(Benchmark):
    def __init__(self, file_path: Path):
        self.df = pd.read_csv(file_path).sample(100)

    def X(self):
        return self.df[["sqft_living", "bedrooms", "bathrooms"]]

    def y(self):
        return self.df[["price"]]

    def _points(self) -> Iterable[Point]:
        return [(row["long"], row["lat"]) for idx, row in self.df.iterrows()]

    def _cache_directory(self) -> Path:
        return Path("data/cache/benchmarks/house_price")


def evaluate(benchmark: Benchmark, seed=1):
    def _score(X_train, X_test, y_train, y_test):

        m = ElasticNetCV(cv=5)
        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)
        return r2_score(y_test, y_pred)

    X_train, X_test, y_train, y_test, embeddings_train, embeddings_test = train_test_split(
        benchmark.X(), benchmark.y(), benchmark.embeddings(),
        test_size=0.2,
        random_state=seed
    )
    print(X_train.shape)
    print(y_train.shape)
    print(embeddings_train.shape)

    print("Testing performance of X")
    X_score = _score(X_train, X_test, y_train, y_test)

    print("Testing performance of embeddings")
    embeddings_score = _score(embeddings_train, embeddings_test, y_train, y_test)

    print("Testing performance of X and embeddings")
    combined_train = np.concatenate([embeddings_train, X_train], axis=1)
    combined_test = np.concatenate([embeddings_test, X_test], axis=1)
    combined_score = _score(combined_train, combined_test, y_train, y_test)

    print(X_score, embeddings_score, combined_score)


def run():
    random.seed()

    evaluate(
        HousePriceBenchmark(
            Path("data/benchmarks/house_prices/data.csv")
        )
    )


if __name__ == "__main__":
    run()
