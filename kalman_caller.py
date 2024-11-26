import csv
import json
import os
import shutil
import ee
import ee.geometry
import numpy as np
import pandas as pd
from kalman.kalman_helper import parse_band_names
from lib.constants import (
    DATE_LABEL,
    FORWARD_TREND_LABEL,
    HARMONIC_FLAGS_LABEL,
    HARMONIC_TREND_LABEL,
    RETROFITTED_TREND_LABEL,
    Index,
    Initialization,
    Kalman,
    Sensor,
)
from lib.study_areas import PNW, RANDONIA
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
    TIMESTAMP_LABEL,
)
from lib.paths import (
    HARMONIC_TREND_SUBDIRECTORY,
    KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY,
    KALMAN_STATE_SUBDIRECTORY,
    build_end_of_year_kalman_state_path,
    build_kalman_analysis_path,
    build_harmonic_trend_path,
    build_kalman_result_path,
    build_kalman_run_directory,
    build_points_path,
    get_kalman_parameters_path,
    kalman_analysis_directory,
    kalman_result_directory,
)
from kalman.kalman_helper import parse_harmonic_params
from kalman.kalman_module import main as eeek
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots
import threading

# could be omitted
RUN_ID = ""

# Parameters
COLLECTION_PARAMETERS = {
    "index": Index.SWIR,
    "sensors": [Sensor.L7, Sensor.L8, Sensor.L9],
    "years": range(2017, 2020),
    "point_group": "randonia_4",
    "study_area": RANDONIA,
    "day_step_size": 4,
    "start_doy": 1,
    "end_doy": 365,
    "cloud_cover_threshold": 20,
    "initialization": Initialization.POSTHOC,
}

HARMONIC_FLAGS = {
    Harmonic.INTERCEPT.value: True,
    Harmonic.SLOPE.value: True,
    Harmonic.UNIMODAL.value: True,
    # Harmonic.BIMODAL.value: True,
    # Harmonic.TRIMODAL.value: True,
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
run_directory = build_kalman_run_directory(
    script_directory, TAG, HARMONIC_FLAGS, RUN_ID
)

# Path to the parameters file containing the process noise, measurement noise, and initial state covariance
parameters_file_path = get_kalman_parameters_path(script_directory, HARMONIC_FLAGS)


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


def run_kalman(parameters, collection, point, year):
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

    # add date column
    df[DATE_LABEL] = pd.to_datetime(df[TIMESTAMP_LABEL], unit="ms").dt.strftime(
        "%Y-%m-%d"
    )

    # drop rows outside of the collection years
    df = df[pd.to_datetime(df[DATE_LABEL]).dt.year == year]

    return df


def process_point(kalman_parameters, point):
    global run_directory

    index, point = point

    harmonic_trend_coefs_path = build_harmonic_trend_path(run_directory, index)

    result_path = build_kalman_result_path(run_directory, index)

    end_of_year_kalman_state_path = build_end_of_year_kalman_state_path(
        run_directory, index
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

        data = run_kalman(kalman_parameters, collection, point, year)

        update_kalman_parameters_with_last_run(data, year)

        data.to_csv(
            result_path,
            mode="a" if not is_first_year else "w",
            header=True if is_first_year else False,
            index=False,
        )

    for year in COLLECTION_PARAMETERS["years"]:
        process_year(year)


def setup_subdirectories():
    result_dir = kalman_result_directory(run_directory)

    os.makedirs(os.path.join(result_dir, KALMAN_STATE_SUBDIRECTORY), exist_ok=True)

    os.makedirs(
        os.path.join(result_dir, HARMONIC_TREND_SUBDIRECTORY),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(result_dir, KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY),
        exist_ok=True,
    )

    os.makedirs(kalman_analysis_directory(run_directory), exist_ok=True)


def generate_all_plots():
    for point_index in range(len(POINTS)):
        generate_plots(
            data=build_kalman_result_path(run_directory, point_index),
            output_path=build_kalman_analysis_path(run_directory, point_index),
            options={
                PlotType.KALMAN_VS_HARMONIC: {
                    "title": f"Kalman vs Harmonic Trend",
                    HARMONIC_TREND_LABEL: build_harmonic_trend_path(
                        run_directory, point_index
                    ),
                    HARMONIC_FLAGS_LABEL: HARMONIC_FLAGS,
                },
                PlotType.KALMAN_FIT: {
                    "title": "Kalman Fit",
                },
                PlotType.KALMAN_VS_CCDC: {
                    "title": "Kalman vs CCDC",
                },
                PlotType.KALMAN_VS_CCDC_COEFS: {
                    "title": "Kalman vs CCDC Coefficients",
                    HARMONIC_FLAGS_LABEL: HARMONIC_FLAGS,
                },
                PlotType.KALMAN_RETROFITTED: {
                    HARMONIC_FLAGS_LABEL: HARMONIC_FLAGS,
                    Kalman.EOY_STATE.value: build_end_of_year_kalman_state_path(
                        run_directory, point_index
                    ),
                    FORWARD_TREND_LABEL: True,
                    HARMONIC_TREND_LABEL: build_harmonic_trend_path(
                        run_directory, point_index
                    ),
                },
            },
        )


def run_continuous_kalman():
    global run_directory

    # create result, harmonic coefficients, and analysis directories
    setup_subdirectories()

    # save point coordinates to file
    json.dump(POINTS, open(build_points_path(run_directory), "w"))

    # load kalman parameters Q, R, P, X
    kalman_parameters = json.load(open(parameters_file_path))

    threads = []
    for index, point in enumerate(POINTS):
        thread = threading.Thread(
            target=process_point, args=(kalman_parameters.copy(), (index, point))
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    generate_all_plots()


if __name__ == "__main__":

    # Ensure the run directory exists
    os.makedirs(run_directory, exist_ok=True)

    # Copy the parameters file to the run directory
    shutil.copy(
        parameters_file_path,
        os.path.join(run_directory, os.path.basename(parameters_file_path)),
    )

    run_continuous_kalman()
