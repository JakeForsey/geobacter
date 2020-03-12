import asyncio
import os
import threading

from flask import Flask
from flask_restful import reqparse, Api, Resource, abort
import httpx
import torch

from geobacter.inference.util import buffer_point
from geobacter.inference.mapnik import request_extent
from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.inference.datasets.osm import BASE_TRANSFORMS

CHECKPOINT = 'checkpoints/ResNetTriplet-OsmTileDataset-6a2d4729-e31b-4866-a67a-1c708a7aa485_embedding_136.pth'
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
parser.add_argument('token', type=str)
parser.add_argument('lon', type=float, help='lon must be a float.')
parser.add_argument('lat', type=float, help='lat must be a float.')


class Embeddings(Resource):
    def get(self):
        args = parser.parse_args()
        abort_if_token_incorrect(args["token"])
        point = args["lon"], args["lat"]

        async with httpx.AsyncClient() as client:
            image_coroutine = request_extent(
                buffer_point(point, 100.0),
                16,
                client
            )
            loop = asyncio.new_event_loop()
            task = loop.create_task(image_coroutine)
            try:
                image = loop.run_until_complete(task)
            except httpx.exceptions.NetworkError:
                abort(404, message=f"Mapnik server is down.")

        image = BASE_TRANSFORMS(image)
        image = image.cuda()
        with model_semaphore:
            return {
                "embeddings": embedding_model(image.unsqueeze(0)).detach().cpu().numpy().tolist(),
                "checkpoint": CHECKPOINT,
                "lon": point[0],
                "lat": point[1]
            }


api.add_resource(Embeddings, '/embeddings')


if __name__ == '__main__':
    app.run(debug=True)
