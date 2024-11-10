from datetime import datetime
import ee
import pandas as pd
from lib.utils import utils
from lib.utils.date import convert_to_fraction_of_year
from lib.utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from lib.utils.ee.dates import get_timestamps_from_image_collection
import math
import csv
from lib.constants import Index


def harmonic_trend_coefficients(collection, point_coords, index: Index):
    modality = {
        "constant": True,
        "linear": False,
        "unimodal": True,
        "bimodal": False,
        "trimodal": False,
    }

    image_collection = ee.ImageCollection(
        collection.filterBounds(ee.Geometry.Point(point_coords))
    )

    reduced_image_collection_with_harmonics = (
        add_harmonic_bands_via_modality_dictionary(image_collection, modality)
    )

    harmonic_independent_variables = (
        determine_harmonic_independents_via_modality_dictionary(modality)
    )

    harmonic_one_time_regression = fit_harmonic_to_collection(
        reduced_image_collection_with_harmonics,
        index.value,
        harmonic_independent_variables,
    )
    fitted_coefficients = harmonic_one_time_regression["harmonic_trend_coefficients"]

    return fitted_coefficients


def fitted_coefficients(
    points, fitted_coefficiets_filename, collection, years: list[int], index: Index
):
    def get_coefficients_for_point(collection, coords, year):
        request = utils.build_request(coords)
        request["expression"] = harmonic_trend_coefficients(collection, coords, index)
        coefficients = utils.compute_pixels_wrapper(request)

        timestamps = get_timestamps_from_image_collection(collection, year, coords)

        return {
            "intercept": coefficients[0],
            "cos": coefficients[1],
            "sin": coefficients[2],
            "timestamps": timestamps,
        }

    output_list = []
    coefficients_by_point = {}

    # Write fitted coefficients
    with open(fitted_coefficiets_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [
                "point",
                "longitude",
                "latitude",
                *[f"intercept_{year}" for year in years],
                *[f"cos_{year}" for year in years],
                *[f"sin_{year}" for year in years],
            ]
        )
        for i, point in enumerate(points):
            coefficients_by_point[i] = {
                "coordinates": (point[0], point[1]),
            }

            for year in years:
                coefficients_by_point[i][year] = get_coefficients_for_point(
                    collection.filterBounds(ee.Geometry.Point(point[0], point[1])),
                    (point[0], point[1]),
                    year,
                )

            # Write coefficients to the CSV file
            csv_writer.writerow(
                [
                    i,
                    point[0],
                    point[1],
                    *[coefficients_by_point[i][year]["intercept"] for year in years],
                    *[coefficients_by_point[i][year]["cos"] for year in years],
                    *[coefficients_by_point[i][year]["sin"] for year in years],
                ]
            )

            output_list.append(coefficients_by_point[i])

    return output_list


def calculate_harmonic_estimate(coefficients, date):
    date = pd.Timestamp(date)

    fraction_of_year = convert_to_fraction_of_year(date)

    phi = 6.283 * fraction_of_year

    phi_cos = math.cos(phi)
    phi_sin = math.sin(phi)

    return (
        coefficients["intercept"]
        + coefficients["cos"] * phi_cos
        + coefficients["sin"] * phi_sin
    )
