from datetime import datetime
import ee
import numpy as np
import pandas as pd
from lib.utils import utils
from lib.utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from lib.utils.ee.dates import get_timestamps_from_image_collection
import csv
from lib.constants import Harmonic, Index


def get_num_sinusoid_pairs(harmonic_flags):
    NUM_SINUSOID_PAIRS = 1

    if harmonic_flags.get(Harmonic.BIMODAL.value):
        NUM_SINUSOID_PAIRS *= 2
    if harmonic_flags.get(Harmonic.TRIMODAL.value):
        NUM_SINUSOID_PAIRS *= 3

    return NUM_SINUSOID_PAIRS


def parse_harmonic_params(harmonic_flags):
    param_names = []

    NUM_SINUSOID_PAIRS = get_num_sinusoid_pairs(harmonic_flags)

    if harmonic_flags.get(Harmonic.INTERCEPT.value):
        param_names.append(Harmonic.INTERCEPT.value)
    if harmonic_flags.get(Harmonic.SLOPE.value):
        param_names.append(Harmonic.SLOPE.value)

    if harmonic_flags.get(Harmonic.UNIMODAL.value):
        param_names.extend([Harmonic.COS.value, Harmonic.SIN.value])

    if harmonic_flags.get(Harmonic.BIMODAL.value):
        param_names.extend([Harmonic.COS2.value, Harmonic.SIN2.value])

    if harmonic_flags.get(Harmonic.TRIMODAL.value):
        param_names.extend([Harmonic.COS3.value, Harmonic.SIN3.value])

    return param_names, NUM_SINUSOID_PAIRS


def harmonic_trend_coefficients(
    collection,
    point_coords,
    years: list[int],
    index: Index,
    harmonic_flags={Harmonic.INTERCEPT: True, Harmonic.UNIMODAL: True},
):
    modality = {
        "constant": harmonic_flags.get(Harmonic.INTERCEPT.value, False),
        "linear": harmonic_flags.get(Harmonic.SLOPE.value, False),
        "unimodal": harmonic_flags.get(Harmonic.UNIMODAL.value, False),
        "bimodal": harmonic_flags.get(Harmonic.BIMODAL.value, False),
        "trimodal": harmonic_flags.get(Harmonic.TRIMODAL.value, False),
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
    harmonic_flags={Harmonic.INTERCEPT: True, Harmonic.UNIMODAL: True},
    output_file: str = None,
):
    coefficients = utils.get_pixels(
        point_coords,
        harmonic_trend_coefficients(
            collection, point_coords, [year], index, harmonic_flags=harmonic_flags
        ),
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

    y = float(coefficients.get(Harmonic.INTERCEPT.value, 0))

    if coefficients.get(Harmonic.SLOPE.value, None):
        y += float(coefficients.get(Harmonic.SLOPE.value, 0)) * frac_of_year

    if coefficients.get(Harmonic.COS.value, None) and coefficients.get(
        Harmonic.SIN.value, None
    ):
        y += float(coefficients.get(Harmonic.COS.value, 0)) * phi_cos
        y += float(coefficients.get(Harmonic.SIN.value, 0)) * phi_sin

    if coefficients.get(Harmonic.COS2.value, None) and coefficients.get(
        Harmonic.SIN2.value, None
    ):
        y += float(coefficients.get(Harmonic.COS2.value, 0)) * np.cos(2 * phi)
        y += float(coefficients.get(Harmonic.SIN2.value, 0)) * np.sin(2 * phi)

    if coefficients.get(Harmonic.COS3.value, None) and coefficients.get(
        Harmonic.SIN3.value, None
    ):
        y += float(coefficients.get(Harmonic.COS3.value, 0)) * np.cos(3 * phi)
        y += float(coefficients.get(Harmonic.SIN3.value, 0)) * np.sin(3 * phi)

    return y
