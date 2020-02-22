for f in data/osm/*.osm.pbf
do
  echo "Processing $(pwd)$f"
  docker run \
      -v $(pwd)/$f:/data.osm.pbf \
      -v openstreetmap-data:/var/lib/postgresql/12/main \
      overv/openstreetmap-tile-server \
      import
done

