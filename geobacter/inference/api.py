import asyncio
import os
from pathlib import Path
import threading

from flask import Flask
from flask_restful import reqparse, Api, Resource, abort
import httpx
import numpy as np
import torch

from geobacter.inference.geotypes import Point
from geobacter.inference.util import buffer_point
from geobacter.inference.mapnik import get_extent
from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.datasets.osm import BASE_TRANSFORMS

CACHE_DIR = Path("data/cache")
CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-c448224c-a38e-4c02-8b8c-572ff00e21db_embedding_45297.pth'
embedding_model = ResNetEmbedding(16, pretrained=False)
embedding_model.load_state_dict(torch.load(CHECKPOINT))
embedding_model.zero_grad()
embedding_model.eval()
embedding_model.cuda()
model_semaphore = threading.BoundedSemaphore(1)
app = Flask(__name__)
api = Api(app)


def abort_if_token_incorrect(token: str):
    if token != os.getenv("GEOBACTER_TOKEN"):
        abort(401, message=f"Invalid token ({token}).")


parser = reqparse.RequestParser()
parser.add_argument('token', type=str, help='provide authentication token.', required=True)
parser.add_argument('lon', type=float, help='lon must be a float.', required=True)
parser.add_argument('lat', type=float, help='lat must be a float.', required=True)


class Embeddings(Resource):
    def get(self):
        args = parser.parse_args()
        abort_if_token_incorrect(args["token"])
        point = args["lon"], args["lat"]

        async def point_to_embedding(point: Point) -> np.ndarray:
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

        loop = asyncio.new_event_loop()
        task = loop.create_task(point_to_embedding(point))
        embeddings = loop.run_until_complete(task)

        with model_semaphore:
            return {
                "embeddings": embeddings.tolist(),
                "checkpoint": CHECKPOINT,
                "lon": point[0],
                "lat": point[1]
            }


api.add_resource(Embeddings, '/embeddings')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=False)
