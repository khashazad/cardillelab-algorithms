import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lib.image_collections import COLLECTIONS
from json import load
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(f"{current_directory}/points/points-filtered.json", "r") as file:
    points = load(file)["points"]

geometries = ee.FeatureCollection(
    ee.List(points).map(lambda coordinate: ee.Feature(ee.Geometry.Point(coordinate)))
)


def process_image(image):
    img = image

    def sample_and_copy(feature):
        return feature.copyProperties(
            img, img.propertyNames().remove(ee.String("nominalDate"))
        )

    sampled = img.sampleRegions(collection=geometries, scale=10, geometries=True).map(
        sample_and_copy
    )

    return sampled


measurements = ee.FeatureCollection(
    COLLECTIONS["L8_L9_2022_2023"].map(process_image).flatten()
)

features = measurements.getInfo()["features"]

measurements = pd.DataFrame(
    [
        {
            "longitude": feature["geometry"]["coordinates"][0],
            "latitude": feature["geometry"]["coordinates"][1],
            "date": feature["properties"]["millis"],
            "swir": feature["properties"]["swir"],
        }
        for feature in features
        if "swir" in feature["properties"]
    ]
)

grouped_measurements = measurements.groupby(["longitude", "latitude"])

for point, data in grouped_measurements:
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))  # Create a figure with two subplots

    data["date"] = pd.to_datetime(data["date"], unit="ms")

    # Plot on the first subplot
    axs[0].scatter(data["date"], data["swir"], label="SWIR", color="red")
    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].set_title(f"{data['longitude'].iloc[0]},{data['latitude'].iloc[0]}")
    axs[0].legend()

    # Plot on the second subplot with a fixed y-axis range
    axs[1].scatter(data["date"], data["swir"], label="SWIR", color="blue")
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].set_ylim(0, 0.5)  # Set fixed y-axis range
    axs[1].set_title(f"{data['longitude'].iloc[0]},{data['latitude'].iloc[0]}")
    axs[1].legend()

    fig.savefig(f"{current_directory}/points/v1/stable/{point}.png")
