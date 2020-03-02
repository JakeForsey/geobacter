
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
```
