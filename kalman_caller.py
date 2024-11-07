import os
import shutil
import ee
import ee.geometry
import pandas as pd
import math
from lib.image_collections import COLLECTIONS
from lib.utils.ee.dates import get_timestamps_from_image_collection
from lib.utils.visualization.charts import (
    ChartType,
    generate_charts_single_run,
)
from lib.utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from lib.utils import utils
from lib.utils.ee import dates as date_utils
from lib.utils.ee import ccdc_utils
from datetime import datetime
from kalman.kalman_helper import main as eeek
import csv
from pprint import pprint
import concurrent.futures
from lib.observations_points import STUDY_POINT_GROUPS, parse_point_coordinates

# Parameters

# the points that are for which the kalman process is run
STUDY_GROUP_TAG = "randonia_4"
POINTS = parse_point_coordinates(STUDY_POINT_GROUPS[STUDY_GROUP_TAG])

# the image collection that is used for the kalman process
COLLECTION_TAG = "Randonia_l8_l9_2017_2018_swir"
COLLECTION = COLLECTIONS[COLLECTION_TAG]

# the years included in the kalman process
YEARS = [2017, 2018]

# whether to include the ccdc coefficients in the output
INCLUDE_CCDC_COEFFICIENTS = True

# the number of sinusoid pairs used in the kalman process
SINUSOID_PAIRS = 1

# the flags for the kalman process
KALMAN_FLAGS = {
    "include_intercept": True,
    "store_measurement": True,
    "store_estimate": True,
    "store_date": True,
    "include_slope": False,
    "store_amplitude": False,
}

# Get the directory of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Define the run directory based on the current timestamp
run_directory = f"{script_directory}/tests/kalman/{STUDY_GROUP_TAG} - {len(STUDY_GROUP_TAG)} points/{datetime.now().strftime('%m-%d %H:%M')}/"

# Path to the parameters file containing the process noise, measurement noise, and initial state covariance
parameters_file_path = f"{script_directory}/kalman/eeek_input.csv"

# Path to the point set directory
point_set_directory_path = f"{script_directory}/points/sets/{STUDY_GROUP_TAG}"

# Create the run directory if it doesn't exist
os.makedirs(run_directory, exist_ok=True)


def append_ccdc_coefficients(kalman_output_path, points):
    ccdc_asset = COLLECTIONS["CCDC_Randonia"]
    bands = ["SWIR1"]
    coefs = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
    segments_count = 10
    segments = ccdc_utils.build_segment_tag(segments_count)

    kalman_output_df = pd.read_csv(kalman_output_path)
    dates = kalman_output_df[kalman_output_df["point"] == 0]["date"].to_numpy()

    ccdc_image = ccdc_utils.build_ccd_image(ccdc_asset, segments_count, bands)

    ccdc_coefs_ic = ee.ImageCollection([])

    for date in dates:
        formatted_date = date_utils.convert_date(
            {"input_format": 2, "input_date": date, "output_format": 1}
        )

        ccdc_coefs_image = ccdc_utils.get_multi_coefs(
            ccdc_image, formatted_date, bands, coefs
        )

        ccdc_coefs_ic = ccdc_coefs_ic.merge(ee.ImageCollection([ccdc_coefs_image]))

    def process_point(i, point):
        coords = (point[0], point[1])
        request = utils.build_request(coords)
        request["expression"] = ccdc_coefs_ic.toBands()
        coef_list = utils.compute_pixels_wrapper(request)

        ccdc_coefs_df = pd.DataFrame(
            coef_list.reshape(len(dates), len(coefs)),
            columns=[f"CCDC_{c}" for c in coefs],
        )
        ccdc_coefs_df["date"] = dates

        ccdc_coefs_df["formatted_date"] = pd.to_datetime(
            ccdc_coefs_df["date"], unit="ms"
        ).dt.strftime("%Y-%m-%d")

        kalman_df = pd.DataFrame(kalman_output_df[kalman_output_df["point"] == i])

        # Ensure both DataFrames have the same number of rows before concatenation
        kalman_ccdc_df = pd.merge(
            kalman_df,
            ccdc_coefs_df,
            on="date",
            how="inner",
        )
        # Save to a temporary CSV file
        temp_file_path = f"{os.path.dirname(kalman_output_path)}/temp_output_{i}.csv"
        kalman_ccdc_df.to_csv(temp_file_path, index=False)
        return temp_file_path

    with concurrent.futures.ThreadPoolExecutor() as executor:
        temp_files = list(executor.map(lambda p: process_point(*p), enumerate(points)))

    output_df = pd.concat(
        [pd.read_csv(temp_file) for temp_file in temp_files if temp_file is not None],
        ignore_index=True,
    )

    output_file_path = kalman_output_path.replace(".csv", "_ccdc.csv")
    output_df.to_csv(output_file_path, index=False)

    # clean up temporary files
    for temp_file in temp_files:
        if temp_file is not None:
            os.remove(temp_file)


def run_kalman():
    global run_directory

    # Ensure the run directory exists
    os.makedirs(run_directory, exist_ok=True)

    # Copy the parameters file to the run directory
    shutil.copy(parameters_file_path, os.path.join(run_directory, "eeek_input.csv"))

    # Define file paths for input and output
    input_file_path = os.path.join(run_directory, "eeek_input.csv")
    output_file_path = os.path.join(run_directory, "eeek_output.csv")
    points_file_path = os.path.join(run_directory, "points.csv")

    # Set up arguments for the Kalman process
    args = {
        "input": input_file_path,
        "output": output_file_path,
        "points": points_file_path,
        "num_sinusoid_pairs": SINUSOID_PAIRS,
        "collection": COLLECTION_TAG,
        "include_intercept": True,
        "store_measurement": True,
        "store_estimate": True,
        "store_date": True,
        "include_slope": False,
        "store_amplitude": False,
    }

    # Run the Kalman process with the specified arguments
    eeek(args)


def harmonic_trend_coefficients(collection, coords):
    modality = {
        "constant": True,
        "linear": False,
        "unimodal": True,
        "bimodal": False,
        "trimodal": False,
    }

    image_collection = ee.ImageCollection(
        collection.filterBounds(ee.Geometry.Point(coords))
    )

    reduced_image_collection_with_harmonics = (
        add_harmonic_bands_via_modality_dictionary(image_collection, modality)
    )

    harmonic_independent_variables = (
        determine_harmonic_independents_via_modality_dictionary(modality)
    )

    harmonic_one_time_regression = fit_harmonic_to_collection(
        reduced_image_collection_with_harmonics, "swir", harmonic_independent_variables
    )
    fitted_coefficients = harmonic_one_time_regression["harmonic_trend_coefficients"]

    return fitted_coefficients


def get_fitted_coefficients_for_point(collection, coords, year):
    request = utils.build_request(coords)
    request["expression"] = harmonic_trend_coefficients(collection, coords)
    coefficients = utils.compute_pixels_wrapper(request)

    image_dates = get_timestamps_from_image_collection(collection, year, coords)

    return {
        "intercept": coefficients[0],
        "cos": coefficients[1],
        "sin": coefficients[2],
        "dates": image_dates,
    }


def fitted_coefficients_and_dates(points, fitted_coefficiets_filename):
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
                *[f"intercept_{year}" for year in YEARS],
                *[f"cos_{year}" for year in YEARS],
                *[f"sin_{year}" for year in YEARS],
            ]
        )
        for i, point in enumerate(points):
            coefficients_by_point[i] = {
                "coordinates": (point[0], point[1]),
            }

            for year in YEARS:
                coefficients_by_point[i][year] = get_fitted_coefficients_for_point(
                    COLLECTION.filterBounds(ee.Geometry.Point(point[0], point[1])),
                    (point[0], point[1]),
                    year,
                )

            # Write coefficients to the CSV file
            csv_writer.writerow(
                [
                    i,
                    point[0],
                    point[1],
                    *[coefficients_by_point[i][year]["intercept"] for year in YEARS],
                    *[coefficients_by_point[i][year]["cos"] for year in YEARS],
                    *[coefficients_by_point[i][year]["sin"] for year in YEARS],
                ]
            )

            output_list.append(coefficients_by_point[i])

    return output_list


def create_points_file(points_filename, coefficients_by_point):
    with open(points_filename, "w", newline="") as file:
        for _, point in enumerate(coefficients_by_point):
            longitude = point["coordinates"][0]
            latitude = point["coordinates"][1]

            year = YEARS[0]

            intercept = point[year]["intercept"]
            cos = point[year]["cos"]
            sin = point[year]["sin"]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")


def build_observations(coefficients_by_point, output_filename):
    observations = []

    with open(output_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["point", "date", "intercept", "cos", "sin", "estimate"])
        for index, dic in enumerate(coefficients_by_point):
            observation_index = 1

            def create_observation_from_coefficients(dates, intercept, cos, sin):
                nonlocal observation_index
                for date in dates:
                    time = (
                        pd.Timestamp(date, unit="ms") - pd.Timestamp("2016-01-01")
                    ).total_seconds() / (365.25 * 24 * 60 * 60)

                    phi = 6.283 * time

                    phi_cos = math.cos(phi)
                    phi_sin = math.sin(phi)

                    estimate = intercept + cos * phi_cos + sin * phi_sin

                    observations.append(
                        (f"intercept_{int(index)}_{observation_index}", intercept)
                    )
                    observations.append((f"cos_{int(index)}_{observation_index}", cos))
                    observations.append((f"sin_{int(index)}_{observation_index}", sin))
                    observations.append(
                        (f"estimate_{int(index)}_{observation_index}", estimate)
                    )

                    csv_writer.writerow([index, date, intercept, cos, sin, estimate])
                    observation_index += 1

            for year in YEARS:
                coefficients = dic[year]
                create_observation_from_coefficients(
                    coefficients["dates"],
                    coefficients["intercept"],
                    coefficients["cos"],
                    coefficients["sin"],
                )

        return observations


if __name__ == "__main__":
    # Define filenames for output
    fitted_coefficiets_filename = run_directory + "fitted_coefficients.csv"
    points_filename = run_directory + "points.csv"
    observations_filename = run_directory + "observations.csv"

    # Check if fitted coefficients file exists, if not, create it
    if not os.path.exists(fitted_coefficiets_filename):
        fitted_coefficiets_by_point = fitted_coefficients_and_dates(
            POINTS, fitted_coefficiets_filename
        )

    # Check if points file exists, if not, create it
    if not os.path.exists(points_filename):
        create_points_file(points_filename, fitted_coefficiets_by_point)

    # Check if observations file exists, if not, create it
    if not os.path.exists(observations_filename):
        observations = build_observations(
            fitted_coefficiets_by_point, observations_filename
        )

    # Run the Kalman process
    run_kalman()

    if INCLUDE_CCDC_COEFFICIENTS:
        append_ccdc_coefficients(f"{run_directory}/eeek_output.csv", POINTS)

    # Generate charts based on the output and observations
    generate_charts_single_run(
        data_file_path=(
            f"{run_directory}/eeek_output_ccdc.csv"
            if INCLUDE_CCDC_COEFFICIENTS
            else f"{run_directory}/eeek_output.csv"
        ),
        observation_file_path=f"{run_directory}/observations.csv",
        output_directory=f"{run_directory}/analysis",
        flags={
            ChartType.KALMAN_VS_HARMONIC_FIT: True,
            ChartType.ESTIMATES_INTERCEPT_COS_SIN: True,
            ChartType.RESIDUALS_OVER_TIME: True,
            ChartType.AMPLITUDE: False,
            ChartType.KALMAN_VS_CCDC: INCLUDE_CCDC_COEFFICIENTS,
        },
    )
