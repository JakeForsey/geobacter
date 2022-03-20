#!/usr/bin/env python
"""
Example of using the API to characterise a few coordinates.
"""
import folium
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Coords created using the online tool: https://www.maps.ie/map-my-route/
COORDS = """6.129277,49.605404 6.131066,49.603605 6.132826,49.601575 6.133297,49.59985 6.132179,49.59782 6.131578,49.595956 6.134544,49.595205 6.136646,49.594927 6.138061,49.594204 6.13965,49.592952 6.140507,49.591394 6.141408,49.589836 6.142309,49.588278 6.143211,49.58672 6.144627,49.585329 6.146688,49.58391 6.148061,49.582352 6.149563,49.580738 6.151409,49.579319 6.153512,49.579541 6.157877,49.582952 6.162676,49.588739 6.170061,49.595415 6.173489,49.601757 6.173141,49.607653 6.176922,49.61277 6.179496,49.617886 6.18104,49.624447 6.181897,49.631008 6.182755,49.636678 6.188427,49.640458 6.19667,49.641792 6.208863,49.642792 6.218645,49.642681 6.228429,49.640791 6.237183,49.638123 6.248,49.637123 6.254907,49.638451 6.262567,49.639979 6.270034,49.641536 6.277115,49.642425 6.281446,49.642786 6.286725,49.643148"""


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


if __name__ == "__main__":
    coords = COORDS.split(" ")
    coords = [coord.split(",") for coord in coords]
    coords = [(float(lat), float(lon)) for lon, lat in coords]

    # Fetch all the embeddings for each coordinate in the spiral
    embeddings = []
    for lat, lon in coords:
        response = requests.get(f"http://localhost:8000/embeddings?lon={lon}&lat={lat}&token=abc")
        embeddings.append(response.json()["embeddings"])
    
    # Reduce the embeddings to 3 dimensions so they can be used as RGB colours
    colours = PCA(n_components=3).fit_transform(embeddings)
    colours = MinMaxScaler(feature_range=(0, 255)).fit_transform(colours).astype(int)

    # Plot the coordinates as circles with the dervice colour
    map = folium.Map(
        location=coords[0],
        zoom_start=15
    )
    for (lat, lon), colour in zip(coords, colours):
        folium.Circle(
            location=[lat, lon],
            weight=0,
            fill_color=rgb_to_hex(tuple(colour.tolist())),
            fill_opacity=0.8,
            radius=50,
        ).add_to(map)

    map.save(f"map.html")
