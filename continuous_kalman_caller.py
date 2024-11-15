import csv
import json
import os
import shutil
import ee
import ee.geometry
import pandas as pd
from kalman.kalman_helper import parse_band_names
from lib.constants import NUM_MEASURES, Index, Initialization, Kalman, Sensor
from lib.study_areas import PNW
from lib.study_packages import (
    get_tag,
    get_points,
    get_collection,
)
from lib.utils.harmonic import (
    calculate_harmonic_estimate,
    harmonic_trend_coefficients_for_year,
)
from lib.utils.visualization.charts import (
    ChartType,
    generate_charts_single_run,
)
from lib.utils import utils
from lib.constants import (
    Harmonic,
    KalmanRecordingFlags,
    ESTIMATE,
    TIMESTAMP,
    POINT_INDEX,
)
from lib.paths import (
    ANALYSIS_DIRECTORY,
    KALMAN_OUTPUT_FILE_PREFIX,
    POINTS_FILE_PREFIX,
    HARMONIC_TREND_COEFS_FILE_PREFIX,
    HARMONIC_TREND_COEFS_DIRECTORY,
    RESULTS_DIRECTORY,
)
from kalman.kalman_helper import parse_harmonic_params

from datetime import datetime
from kalman.kalman_module import main as eeek
from pprint import pprint

# Parameters
COLLECTION_PARAMETERS = {
    "index": Index.SWIR,
    "sensors": [Sensor.L7, Sensor.L8],
    "years": [2012, 2013, 2014, 2015, 2016],
    "point_group": "pnw_1",
    "study_area": PNW,
    "day_step_size": 6,
    "start_doy": 1,
    "end_doy": 300,
    "cloud_cover_threshold": 20,
    "initialization": Initialization.POSTHOC,
}

HARMONIC_FLAGS = {
    Harmonic.INTERCEPT.value: True,
    Harmonic.SLOPE.value: False,
    Harmonic.UNIMODAL.value: True,
    Harmonic.BIMODAL.value: False,
    Harmonic.TRIMODAL.value: False,
}

TAG = get_tag(**COLLECTION_PARAMETERS)
POINTS = get_points(COLLECTION_PARAMETERS["point_group"])
INDEX = COLLECTION_PARAMETERS["index"]

# whether to include the ccdc coefficients in the output
INCLUDE_CCDC_COEFFICIENTS = False
INITIALIZATION = Initialization.POSTHOC

# Get the directory of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Define the run directory based on the current timestamp
run_directory = (
    f"{script_directory}/tests/kalman/{TAG}/{datetime.now().strftime('%m-%d %H:%M')}/"
)

# Path to the parameters file containing the process noise, measurement noise, and initial state covariance
parameters_file_path = f"{script_directory}/kalman/kalman_parameters.json"

# Create the run directory if it doesn't exist

# Ensure the run directory exists
os.makedirs(run_directory, exist_ok=True)

# Copy the parameters file to the run directory
shutil.copy(
    parameters_file_path,
    os.path.join(run_directory, os.path.basename(parameters_file_path)),
)


def create_points_file(points_filename, coefficients_by_point, years: list[int]):
    with open(points_filename, "w", newline="") as file:
        for _, coefs in enumerate(coefficients_by_point):
            longitude = coefs["coordinates"][0]
            latitude = coefs["coordinates"][1]

            year = years[0]

            intercept = coefs[year][Harmonic.INTERCEPT.value]
            cos = coefs[year][Harmonic.COS.value]
            sin = coefs[year][Harmonic.SIN.value]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")


def build_observations(coefficients_by_point, output_filename, years: list[int]):
    observations = []

    with open(output_filename, "a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [
                POINT_INDEX,
                TIMESTAMP,
                Harmonic.INTERCEPT.value,
                Harmonic.COS.value,
                Harmonic.SIN.value,
                ESTIMATE,
            ]
        )
        for index, coefficient_set in enumerate(coefficients_by_point):
            observation_index = 1

            def create_observation_from_coefficients(timestamps, intercept, cos, sin):
                nonlocal observation_index
                for timestamp in timestamps:
                    estimate = calculate_harmonic_estimate(
                        {
                            Harmonic.INTERCEPT.value: intercept,
                            Harmonic.COS.value: cos,
                            Harmonic.SIN.value: sin,
                        },
                        timestamp,
                    )

                    observations.append(
                        (
                            f"{Harmonic.INTERCEPT.value}_{int(index)}_{observation_index}",
                            intercept,
                        )
                    )
                    observations.append(
                        (f"{Harmonic.COS.value}_{int(index)}_{observation_index}", cos)
                    )
                    observations.append(
                        (f"{Harmonic.SIN.value}_{int(index)}_{observation_index}", sin)
                    )
                    observations.append(
                        (
                            f"{ESTIMATE}_{int(index)}_{observation_index}",
                            estimate,
                        )
                    )

                    csv_writer.writerow(
                        [index, timestamp, intercept, cos, sin, estimate]
                    )
                    observation_index += 1

            for year in years:
                coefficients = coefficient_set[year]
                create_observation_from_coefficients(
                    coefficients[TIMESTAMP],
                    coefficients[Harmonic.INTERCEPT.value],
                    coefficients[Harmonic.COS.value],
                    coefficients[Harmonic.SIN.value],
                )

        return observations


def run_kalman(parameters, collection, point):
    global run_directory

    collection = collection.filterBounds(ee.Geometry.Point(point))

    # Set up arguments for the Kalman process
    args = {
        "kalman_parameters": parameters,
        "value_collection": collection,
        "harmonic_flags": HARMONIC_FLAGS,
        "recording_flags": {
            KalmanRecordingFlags.MEASUREMENT: True,
            KalmanRecordingFlags.TIMESTAMP: True,
            KalmanRecordingFlags.ESTIMATE: True,
            KalmanRecordingFlags.AMPLITUDE: False,
            KalmanRecordingFlags.STATE: True,
            KalmanRecordingFlags.STATE_COV: True,
            KalmanRecordingFlags.CCDC_COEFFICIENTS: INCLUDE_CCDC_COEFFICIENTS,
        },
    }

    band_names = parse_band_names(args["recording_flags"], args["harmonic_flags"])

    # call the kalman filter
    states = eeek(**args)

    # process the output
    data = utils.get_pixels(point, states).reshape(-1, len(band_names))

    df = pd.DataFrame(data, columns=band_names)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")

    return df


def update_kalman_parameters_with_last_run(kalman_parameters, data):
    harmonic_params, _ = parse_harmonic_params(HARMONIC_FLAGS)

    kalman_parameters[Kalman.X.value] = data.iloc[-1][harmonic_params].tolist()

    kalman_parameters[Kalman.P.value] = data.iloc[-1][
        [f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params]
    ].tolist()


def process_point(kalman_parameters, point):
    global run_directory

    index, point = point

    harmonic_trend_coefs_path = os.path.join(
        run_directory,
        HARMONIC_TREND_COEFS_DIRECTORY,
        f"{HARMONIC_TREND_COEFS_FILE_PREFIX}_{index}.csv",
    )

    result_path = os.path.join(
        run_directory, RESULTS_DIRECTORY, f"{KALMAN_OUTPUT_FILE_PREFIX}_{index}.csv"
    )

    def process_year(year):
        collection = get_collection(
            **{
                **COLLECTION_PARAMETERS,
                "years": [year],
            }
        )

        coefficients = harmonic_trend_coefficients_for_year(
            collection,
            point,
            year,
            INDEX,
            output_file=harmonic_trend_coefs_path,
        )

        if (
            INITIALIZATION == Initialization.POSTHOC
            and year == COLLECTION_PARAMETERS["years"][0]
        ):
            kalman_parameters[Kalman.X.value] = coefficients

        data = run_kalman(kalman_parameters, collection, point)

        update_kalman_parameters_with_last_run(kalman_parameters, data)

        data.to_csv(result_path, mode="a", index=False)

    for year in COLLECTION_PARAMETERS["years"]:
        process_year(year)


def post_run_processing(kalman_parameters, point):
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


def setup_subdirectories():
    os.makedirs(
        os.path.join(run_directory, HARMONIC_TREND_COEFS_DIRECTORY), exist_ok=True
    )

    os.makedirs(os.path.join(run_directory, RESULTS_DIRECTORY), exist_ok=True)

    os.makedirs(os.path.join(run_directory, ANALYSIS_DIRECTORY), exist_ok=True)


def run_continuous_kalman():
    global run_directory

    # create result, harmonic coefficients, and analysis directories
    setup_subdirectories()

    # save point coordinates to file
    json.dump(POINTS, open(os.path.join(run_directory, POINTS_FILE_PREFIX), "w"))

    # load kalman parameters Q, R, P, X
    kalman_parameters = json.load(open(parameters_file_path))

    for index, point in enumerate(POINTS):
        process_point(kalman_parameters.copy(), (index, point))

    # with ProcessPool(nodes=40) as pool:
    # pool.map(process_point, kalman_parameters, POINTS)


if __name__ == "__main__":
    run_continuous_kalman()
