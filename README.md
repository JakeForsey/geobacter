
Set up Mapnik tile server

```
docker-compose build mapnik
docker volume create openstreetmap-data
import_osm.sh 
```

Add additional osm.pbf files:

```
docker run \
    -v /absolute/path/to/additional.osm.pbf:/data.osm.pbf \
    -v openstreetmap-data:/var/lib/postgresql/12/main \
    overv/openstreetmap-tile-server \
    import
```
