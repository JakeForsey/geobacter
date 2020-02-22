
Add all .osm.pbf files to mapnik server:

```
docker-compose build mapnik

docker volume create openstreetmap-data
import_osm.sh
```