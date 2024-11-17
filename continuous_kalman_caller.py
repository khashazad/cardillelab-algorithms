import csv
import json
import os
import shutil
import ee
import ee.geometry
import numpy as np
import pandas as pd
from kalman.kalman_helper import parse_band_names
from lib.constants import DATE, Index, Initialization, Kalman, Sensor
from lib.study_areas import PNW
from lib.study_packages import (
    get_tag,
    get_points,
    get_collection,
)
from lib.utils.harmonic import (
    harmonic_trend_coefficients_for_year,
)
from lib.utils.visualization.plot_generator import generate_plots
from lib.utils import utils
from lib.constants import (
    Harmonic,
    KalmanRecordingFlags,
    TIMESTAMP,
)
from lib.paths import (
    ANALYSIS_DIRECTORY,
    END_OF_YEAR_KALMAN_STATE_FILE_PREFIX,
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

from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots

# Parameters
COLLECTION_PARAMETERS = {
    "index": Index.SWIR,
    "sensors": [Sensor.L8, Sensor.L9],
    "years": list(range(2021, 2022)),
    "point_group": "pnw_1",
    "study_area": PNW,
    "day_step_size": 6,
    "start_doy": 1,
    "end_doy": 365,
    "cloud_cover_threshold": 20,
    "initialization": Initialization.POSTHOC,
}

HARMONIC_FLAGS = {
    Harmonic.INTERCEPT.value: True,
    Harmonic.SLOPE.value: True,
    Harmonic.UNIMODAL.value: True,
    Harmonic.BIMODAL.value: True,
    Harmonic.TRIMODAL.value: False,
}

TAG = get_tag(**COLLECTION_PARAMETERS)
POINTS = get_points(COLLECTION_PARAMETERS["point_group"])
INDEX = COLLECTION_PARAMETERS["index"]
INITIALIZATION = Initialization.POSTHOC

# whether to include the ccdc coefficients in the output
INCLUDE_CCDC_COEFFICIENTS = True

# Get the directory of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Define the run directory based on the current timestamp
run_directory = (
    f"{script_directory}/tests/kalman/{TAG}/{datetime.now().strftime('%m-%d %H:%M')}/"
)

# Path to the parameters file containing the process noise, measurement noise, and initial state covariance
parameters_file_path = f"{script_directory}/kalman/kalman_parameters_slope_bimodal.json"


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
            KalmanRecordingFlags.FRACTION_OF_YEAR: True,
            KalmanRecordingFlags.ESTIMATE: True,
            KalmanRecordingFlags.AMPLITUDE: False,
            KalmanRecordingFlags.STATE: True,
            KalmanRecordingFlags.STATE_COV: True,
            KalmanRecordingFlags.CCDC_COEFFICIENTS: INCLUDE_CCDC_COEFFICIENTS,
        },
    }

    band_names = parse_band_names(args["recording_flags"], args["harmonic_flags"])

    # call the kalman filter
    kalman_output_collection = eeek(**args)

    # process the output
    data = utils.get_image_collection_pixels(point, kalman_output_collection).reshape(
        -1, len(band_names)
    )

    df = pd.DataFrame(data, columns=band_names)
    df[DATE] = pd.to_datetime(df[TIMESTAMP], unit="ms").dt.strftime("%Y-%m-%d")

    return df


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

    end_of_year_kalman_state_path = os.path.join(
        run_directory,
        RESULTS_DIRECTORY,
        f"{END_OF_YEAR_KALMAN_STATE_FILE_PREFIX}_{index}.csv",
    )

    with open(end_of_year_kalman_state_path, "w") as file:
        state_labels, _ = parse_harmonic_params(HARMONIC_FLAGS)
        covariance_labels = [f"{Kalman.COV_PREFIX.value}_{x}" for x in state_labels]

        csv.writer(file).writerow(["year", *state_labels, *covariance_labels])

    def update_kalman_parameters_with_last_run(data, year):
        harmonic_params, _ = parse_harmonic_params(HARMONIC_FLAGS)

        state = data.iloc[-1][harmonic_params].tolist()
        covariance = data.iloc[-1][
            [f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params]
        ].tolist()

        kalman_parameters[Kalman.X.value] = state
        kalman_parameters[Kalman.P.value] = np.diag(covariance)

        with open(end_of_year_kalman_state_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow([year, *state, *covariance])

    def process_year(year):

        is_first_year = year == COLLECTION_PARAMETERS["years"][0]

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
            harmonic_flags=HARMONIC_FLAGS,
            output_file=harmonic_trend_coefs_path,
        )

        if INITIALIZATION == Initialization.POSTHOC and is_first_year:
            kalman_parameters[Kalman.X.value] = coefficients

        data = run_kalman(kalman_parameters, collection, point)

        update_kalman_parameters_with_last_run(data, year)

        data.to_csv(
            result_path,
            mode="a" if not is_first_year else "w",
            header=True if is_first_year else False,
            index=False,
        )

    for year in COLLECTION_PARAMETERS["years"]:
        process_year(year)

    generate_plots(
        data=result_path,
        output_directory=os.path.join(run_directory, ANALYSIS_DIRECTORY, f"{index}"),
        options={
            PlotType.KALMAN_VS_HARMONIC: {
                "title": f"Kalman vs Harmonic Trend",
                "harmonic_trend": harmonic_trend_coefs_path,
                "harmonic_flags": HARMONIC_FLAGS,
            },
            PlotType.KALMAN_FIT: {
                "title": "Kalman Fit",
            },
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

    # Ensure the run directory exists
    os.makedirs(run_directory, exist_ok=True)

    # Copy the parameters file to the run directory
    shutil.copy(
        parameters_file_path,
        os.path.join(run_directory, os.path.basename(parameters_file_path)),
    )

    run_continuous_kalman()
