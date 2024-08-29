import os 
import shutil
import random
from eeek.image_collections import COLLECTIONS
from eeek.utils import build_request, compute_pixels_wrapper
import ee
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

image_collection = COLLECTIONS["L8_L9_2022_2023"]

script_directory = os.path.dirname(os.path.abspath(__file__))
measurement_directory = os.path.join(script_directory,"points", "unstable")

# if os.path.exists(measurement_directory):
#     shutil.rmtree(measurement_directory)
# os.makedirs(measurement_directory)

polygon_coords = [(-126.04, 49.59), (-126.04, 40.76), (-118.93, 40.76), (-118.93, 49.59)]


def generate_random_points(polygon, num_points, scale=10):
    min_lat = min(polygon, key=lambda x: x[1])[1]
    max_lat = max(polygon, key=lambda x: x[1])[1]
    min_lon = min(polygon, key=lambda x: x[0])[0]
    max_lon = max(polygon, key=lambda x: x[0])[0]

    points = []
    for _ in range(num_points):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        
        # lat += random.uniform(-scale * 0.00001, scale * 0.00001)
        # lon += random.uniform(-scale * 0.00001, scale * 0.00001)
        points.append((lon, lat))
    
    return points


valid_points_counter = 0

while valid_points_counter < 5:
    geometries = ee.FeatureCollection(ee.List(generate_random_points(polygon_coords, 100, 10)).map(lambda coordinate: ee.Feature(ee.Geometry.Point(coordinate))))

    def process_image(image):
        img = image

        def sample_and_copy(feature):
            return feature.copyProperties(img, img.propertyNames().remove(ee.String("nominalDate")))

        sampled = img.sampleRegions(
            collection=geometries,
            scale=10,
            geometries=True
        ).map(sample_and_copy)

        return sampled

    measurements = ee.FeatureCollection(
        image_collection.map(process_image).flatten()
    )

    features = measurements.getInfo()["features"]

    measurements = pd.DataFrame([
        {
            "longitude": feature["geometry"]["coordinates"][0],
            "latitude": feature["geometry"]["coordinates"][1],
            "date": feature["properties"]["millis"],
            "swir": feature["properties"]["swir"]
        }
        for feature in features if "swir" in feature["properties"]
    ])

    grouped_measurements = measurements.groupby(['longitude', 'latitude'])

    for point, data in grouped_measurements:

        if (len(data) < 10):
            continue

        data['year'] = pd.to_datetime(data['date'], unit='ms').dt.year
        mean_swir_2022 = data[data['year'] == 2022]['swir'].mean()
        mean_swir_2023 = data[data['year'] == 2023]['swir'].mean()

        if abs(mean_swir_2022 - mean_swir_2023) < 0.2:
            continue
        
        valid_points_counter += 1

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))

        data['date'] = pd.to_datetime(data['date'], unit='ms')

        axs.scatter(data['date'], data['swir'], label='swir', color='red')

        axs.xaxis.set_major_locator(mdates.AutoDateLocator())
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axs.tick_params(axis='x', labelsize=8)

        # axs.set_ylim(0, 0.5)

        axs.set_title(f"{data['longitude'].iloc[0]},{data['latitude'].iloc[0]}")
        axs.legend()


        fig.savefig(f"{measurement_directory}/{point}.png")



