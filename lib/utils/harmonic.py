from datetime import datetime
import ee
import numpy as np
import pandas as pd
from lib.utils import utils
from lib.utils.date import timestamp_to_frac_of_year
from lib.utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from lib.utils.ee.dates import get_timestamps_from_image_collection
import math
import csv
from lib.constants import Harmonic, Index


def harmonic_trend_coefficients(
    collection, point_coords, years: list[int], index: Index
):
    modality = {
        "constant": True,
        "linear": False,
        "unimodal": True,
        "bimodal": False,
        "trimodal": False,
    }

    image_collection = ee.ImageCollection(
        collection.filterBounds(ee.Geometry.Point(point_coords)).filter(
            ee.Filter.calendarRange(years[0], years[-1], "year")
        )
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


def harmonic_trend_coefficients_for_year(
    collection,
    point_coords,
    year: int,
    index: Index,
    output_file: str = None,
):
    coefficients = utils.get_pixels(
        point_coords,
        harmonic_trend_coefficients(collection, point_coords, [year], index),
    )

    if output_file:
        with open(output_file, "a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([year, *coefficients])

    return coefficients


def harmonic_trend_coefficients_for_points(
    points, fitted_coefficiets_filename, collection, years: list[int], index: Index
):
    def get_coefficients_for_point(collection, coords, year):
        coefficients = utils.get_pixels(
            coords, harmonic_trend_coefficients(collection, coords, year, index)
        )

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
    with open(fitted_coefficiets_filename, "a", newline="") as file:
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


def calculate_harmonic_estimate(coefficients, frac_of_year):
    phi = np.pi * 2 * frac_of_year

    phi_cos = np.cos(phi)
    phi_sin = np.sin(phi)

    y = float(coefficients.get(Harmonic.INTERCEPT))

    if coefficients.get(Harmonic.SLOPE):
        y += float(coefficients.get(Harmonic.SLOPE)) * frac_of_year

    if coefficients.get(Harmonic.COS) and coefficients.get(Harmonic.SIN):
        y += (
            float(coefficients.get(Harmonic.COS)) * phi_cos
            + float(coefficients.get(Harmonic.SIN)) * phi_sin
        )

    if coefficients.get(Harmonic.COS1) and coefficients.get(Harmonic.SIN1):
        y += float(coefficients.get(Harmonic.COS1)) * np.cos(2 * phi) + float(
            coefficients.get(Harmonic.SIN1)
        ) * np.sin(2 * phi)

    if coefficients.get(Harmonic.COS2) and coefficients.get(Harmonic.SIN2):
        y += float(coefficients.get(Harmonic.COS2)) * np.cos(3 * phi) + float(
            coefficients.get(Harmonic.SIN2)
        ) * np.sin(3 * phi)

    return y
