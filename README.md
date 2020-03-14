
Setup
```
docker-compose build mapnik
docker volume create openstreetmap-data
docker run \
    -e THREADS=12 \
    -v $PWD/data/osm/great-britain-latest.osm.pbf:/data.osm.pbf \
    -v openstreetmap-data:/var/lib/postgresql/12/main \
    mapnik \
    import

docker volume create openstreetmap-rendered-tiles

docker-compose up
```

Run

```
export GEOBACTER_TOKEN=<token>
docker-compose up
gunicorn -b 0.0.0.0:8000 --workers 4 --timeout 10 geobacter.inference.api:app
```