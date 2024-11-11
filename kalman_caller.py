import csv
import os
import shutil
import ee
import ee.geometry
import pandas as pd
from lib.constants import Index
from lib.image_collections import COLLECTIONS
from lib.study_packages import pnw_nbr_2017_2018_1_point, pnw_nbr_2017_2019_1_point
from lib.utils.harmonic import (
    calculate_harmonic_estimate,
    fitted_coefficients,
)
from lib.utils.visualization.charts import (
    ChartType,
    generate_charts_single_run,
)

from lib.utils import utils
from lib.utils.ee import dates as date_utils
from lib.utils.ee import ccdc_utils
from datetime import datetime
from kalman.kalman_helper import main as eeek
from pprint import pprint
import concurrent.futures

# Parameters
TAG, INDEX, POINTS, COLLECTION, YEARS = pnw_nbr_2017_2018_1_point().values()

# whether to include the ccdc coefficients in the output
INCLUDE_CCDC_COEFFICIENTS = False

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

INITIALIZATION = "uninformative"

# Get the directory of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Define the run directory based on the current timestamp
run_directory = (
    f"{script_directory}/tests/kalman/{TAG}/{datetime.now().strftime('%m-%d %H:%M')}/"
)

# Path to the parameters file containing the process noise, measurement noise, and initial state covariance
parameters_file_path = f"{script_directory}/kalman/eeek_input.json"

# Create the run directory if it doesn't exist
os.makedirs(run_directory, exist_ok=True)


def append_ccdc_coefficients(kalman_output_path, points):
    ccdc_asset = COLLECTIONS["CCDC_Global"]
    bands = ["SWIR1"]
    coefs = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
    segments_count = 10

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

        kalman_df = pd.DataFrame(kalman_output_df[kalman_output_df["point"] == i])

        kalman_ccdc_df = pd.merge(
            kalman_df,
            ccdc_coefs_df,
            on="date",
            how="inner",
        )

        # Save to a temporary CSV file
        temp_file_path = (
            f"{os.path.dirname(kalman_output_path)}/outputs/{i:03d}_with_ccdc.csv"
        )
        kalman_ccdc_df.to_csv(temp_file_path, index=False)

        return temp_file_path

    with concurrent.futures.ThreadPoolExecutor() as executor:
        temp_files = list(executor.map(lambda p: process_point(*p), enumerate(points)))

    output_df = pd.concat(
        [pd.read_csv(temp_file) for temp_file in temp_files if temp_file is not None],
        ignore_index=True,
    )

    output_file_path = kalman_output_path.replace(".csv", "_with_ccdc.csv")
    output_df.to_csv(output_file_path, index=False)

    # clean up temporary files
    # for temp_file in temp_files:
    #     if temp_file is not None:
    #         os.remove(temp_file)


def create_points_file(points_filename, coefficients_by_point, years: list[int]):
    with open(points_filename, "w", newline="") as file:
        for _, coefs in enumerate(coefficients_by_point):
            longitude = coefs["coordinates"][0]
            latitude = coefs["coordinates"][1]

            year = years[0]

            intercept = coefs[year]["intercept"]
            cos = coefs[year]["cos"]
            sin = coefs[year]["sin"]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")


def build_observations(coefficients_by_point, output_filename, years: list[int]):
    observations = []

    with open(output_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["point", "timestamp", "intercept", "cos", "sin", "estimate"]
        )
        for index, coefficient_set in enumerate(coefficients_by_point):
            observation_index = 1

            def create_observation_from_coefficients(timestamps, intercept, cos, sin):
                nonlocal observation_index
                for timestamp in timestamps:
                    estimate = calculate_harmonic_estimate(
                        {
                            "intercept": intercept,
                            "cos": cos,
                            "sin": sin,
                        },
                        timestamp,
                    )

                    observations.append(
                        (f"intercept_{int(index)}_{observation_index}", intercept)
                    )
                    observations.append((f"cos_{int(index)}_{observation_index}", cos))
                    observations.append((f"sin_{int(index)}_{observation_index}", sin))
                    observations.append(
                        (f"estimate_{int(index)}_{observation_index}", estimate)
                    )

                    csv_writer.writerow(
                        [index, timestamp, intercept, cos, sin, estimate]
                    )
                    observation_index += 1

            for year in years:
                coefficients = coefficient_set[year]
                create_observation_from_coefficients(
                    coefficients["timestamps"],
                    coefficients["intercept"],
                    coefficients["cos"],
                    coefficients["sin"],
                )

        return observations


def run_kalman():
    global run_directory

    # Ensure the run directory exists
    os.makedirs(run_directory, exist_ok=True)

    # Copy the parameters file to the run directory
    shutil.copy(
        parameters_file_path,
        os.path.join(run_directory, os.path.basename(parameters_file_path)),
    )

    # Define file paths for input and output
    input_file_path = os.path.join(
        run_directory, os.path.basename(parameters_file_path)
    )
    output_file_path = os.path.join(run_directory, "eeek_output.csv")
    points_file_path = os.path.join(run_directory, "points.csv")

    # Set up arguments for the Kalman process
    args = {
        "input": input_file_path,
        "output": output_file_path,
        "points": points_file_path,
        "num_sinusoid_pairs": SINUSOID_PAIRS,
        "collection": COLLECTION,
        "include_intercept": True,
        "store_measurement": True,
        "store_estimate": True,
        "store_date": True,
        "include_slope": False,
        "store_amplitude": False,
        "individual_outputs": True,
        "initialization": INITIALIZATION,
    }

    # Run the Kalman process with the specified arguments
    eeek(args)


if __name__ == "__main__":
    # Define filenames for output
    fitted_coefficiets_filename = run_directory + "fitted_coefficients.csv"
    points_filename = run_directory + "points.csv"
    observations_filename = run_directory + "observations.csv"

    fitted_coefficiets_by_point = fitted_coefficients(
        POINTS, fitted_coefficiets_filename, COLLECTION, YEARS, INDEX
    )
    create_points_file(points_filename, fitted_coefficiets_by_point, YEARS)

    observations = build_observations(
        fitted_coefficiets_by_point, observations_filename, YEARS
    )

    # Run the Kalman process
    run_kalman()

    if INCLUDE_CCDC_COEFFICIENTS:
        append_ccdc_coefficients(f"{run_directory}/eeek_output.csv", POINTS)

    # Generate charts based on the output and observations
    generate_charts_single_run(
        data_file_path=(
            f"{run_directory}/eeek_output_with_ccdc.csv"
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
