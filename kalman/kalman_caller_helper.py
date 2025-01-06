import csv
import json
import os
import ee
import ee.geometry
import numpy as np
import pandas as pd
from kalman.kalman_helper import parse_band_names
from lib.constants import (
    CCDC,
    DATE_LABEL,
    ESTIMATE_PREDICTED_LABEL,
    FORWARD_TREND_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_FLAGS_LABEL,
    HARMONIC_TREND_LABEL,
    Initialization,
    Kalman,
)
from lib.study_packages import get_collection
from lib.utils.harmonic import harmonic_trend_coefficients_for_year
from lib.utils.visualization.plot_generator import generate_plots
from lib.utils import utils
from lib.constants import (
    Harmonic,
    KalmanRecordingFlags,
    TIMESTAMP_LABEL,
)
from lib.paths import (
    CCDC_SEGMENTS_SUBDIRECTORY,
    HARMONIC_TREND_SUBDIRECTORY,
    KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY,
    KALMAN_STATE_SUBDIRECTORY,
    RESIDUALS_SUBDIRECTORY,
    build_ccdc_segments_path,
    build_end_of_year_kalman_state_path,
    build_kalman_analysis_path,
    build_harmonic_trend_path,
    build_kalman_result_path,
    build_points_path,
    build_residuals_path,
    kalman_analysis_directory,
    kalman_result_directory,
)
from kalman.kalman_helper import parse_harmonic_params
from kalman.kalman_module import main as eeek
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots
import threading
from lib.utils.ee.ccdc_utils import (
    get_ccdc_coefs_for_date,
    get_segments_for_coordinates,
)


def run_kalman(parameters, collection, point, year, harmonic_flags, include_ccdc):
    global run_directory

    collection = collection.filterBounds(ee.Geometry.Point(point))

    # Set up arguments for the Kalman process
    args = {
        "kalman_parameters": parameters,
        "value_collection": collection,
        "harmonic_flags": harmonic_flags,
        "recording_flags": {
            KalmanRecordingFlags.MEASUREMENT: True,
            KalmanRecordingFlags.TIMESTAMP: True,
            KalmanRecordingFlags.FRACTION_OF_YEAR: True,
            KalmanRecordingFlags.ESTIMATE: True,
            KalmanRecordingFlags.ESTIMATE_PREDICTED: True,
            KalmanRecordingFlags.AMPLITUDE: False,
            KalmanRecordingFlags.STATE: True,
            KalmanRecordingFlags.STATE_COV: True,
            KalmanRecordingFlags.CCDC_COEFFICIENTS: include_ccdc,
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


def process_residuals(residuals_path, data, is_first_year):
    data = data[data[Kalman.Z.value] != 0.0]

    data = data[
        [
            ESTIMATE_PREDICTED_LABEL,
            Kalman.Z.value,
            CCDC.FIT.value,
            TIMESTAMP_LABEL,
            FRACTION_OF_YEAR_LABEL,
            DATE_LABEL,
        ]
    ]

    data["kalman_residual"] = data[ESTIMATE_PREDICTED_LABEL] - data[Kalman.Z.value]

    data["kalman_residual_sq"] = data["kalman_residual"] ** 2

    data["ccdc_residual"] = data[CCDC.FIT.value] - data[ESTIMATE_PREDICTED_LABEL]

    data["ccdc_residual_sq"] = data["ccdc_residual"] ** 2

    # Sort the dataframe columns in a specific order
    sorted_columns = [
        ESTIMATE_PREDICTED_LABEL,
        Kalman.Z.value,
        CCDC.FIT.value,
        "kalman_residual",
        "kalman_residual_sq",
        "ccdc_residual",
        "ccdc_residual_sq",
        TIMESTAMP_LABEL,
        DATE_LABEL,
        FRACTION_OF_YEAR_LABEL,
    ]

    data = data[sorted_columns].rename(
        columns={
            "kalman_residual": "kalman_residual",
            "kalman_residual_sq": "kalman_residual_sq",
            "ccdc_residual": "ccdc_residual",
            "ccdc_residual_sq": "ccdc_residual_sq",
            TIMESTAMP_LABEL: "timestamp",
            DATE_LABEL: "date",
            FRACTION_OF_YEAR_LABEL: "frac_of_year",
            ESTIMATE_PREDICTED_LABEL: "kalman_estimate",
            Kalman.Z.value: "observation",
            CCDC.FIT.value: "ccdc_estimate",
        }
    )

    data.to_csv(
        residuals_path,
        mode="a" if not is_first_year else "w",
        header=True if is_first_year else False,
        index=False,
    )


def process_point(
    run_directory,
    kalman_parameters,
    point,
    harmonic_flags,
    collection_parameters,
    initialization,
    include_ccdc,
):
    index, point = point

    harmonic_trend_coefs_path = build_harmonic_trend_path(run_directory, index)

    result_path = build_kalman_result_path(run_directory, index)

    end_of_year_kalman_state_path = build_end_of_year_kalman_state_path(
        run_directory, index
    )

    harmonic_params, num_sinusoid_pairs = parse_harmonic_params(harmonic_flags)

    with open(end_of_year_kalman_state_path, "w") as file:
        covariance_labels = [f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params]

        csv.writer(file).writerow(["year", *harmonic_params, *covariance_labels])

    def update_kalman_parameters_with_last_run(data, year):
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

        is_first_year = year == collection_parameters["years"][0]

        collection = get_collection(
            **{
                **collection_parameters,
                "years": [year],
                "study_area": ee.Geometry.Point(point),
            }
        )

        coefficients = harmonic_trend_coefficients_for_year(
            collection,
            point,
            year,
            collection_parameters["index"],
            harmonic_flags=harmonic_flags,
            output_file=harmonic_trend_coefs_path,
        )

        if is_first_year:
            if initialization == Initialization.POSTHOC:
                kalman_parameters[Kalman.X.value] = coefficients
            elif initialization == Initialization.CCDC:
                ccdc_coefs = get_ccdc_coefs_for_date(collection.first().date().millis())

                coefs = utils.get_pixels(point, ccdc_coefs)

                state = []

                if harmonic_flags.get(Harmonic.INTERCEPT.value, False):
                    state.append(coefs[0])

                if harmonic_flags.get(Harmonic.SLOPE.value, False):
                    state.append(coefs[1])

                if num_sinusoid_pairs >= 1:
                    state.extend([coefs[2], coefs[3]])

                if num_sinusoid_pairs >= 2:
                    state.extend([coefs[4], coefs[5]])

                if num_sinusoid_pairs >= 3:
                    state.extend([coefs[6], coefs[7]])

                kalman_parameters[Kalman.X.value] = state

        data = run_kalman(
            kalman_parameters,
            collection,
            point,
            year,
            harmonic_flags,
            include_ccdc,
        )

        update_kalman_parameters_with_last_run(data, year)

        if include_ccdc:
            process_residuals(
                build_residuals_path(run_directory, index),
                data.copy(deep=True),
                is_first_year,
            )

        data.to_csv(
            result_path,
            mode="a" if not is_first_year else "w",
            header=True if is_first_year else False,
            index=False,
        )

    for year in collection_parameters["years"]:
        process_year(year)

    if include_ccdc:
        segments_path = build_ccdc_segments_path(run_directory, index)

        segments = get_segments_for_coordinates(point)

        json.dump(segments, open(segments_path, "w"))


def setup_subdirectories(run_directory):
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

    os.makedirs(os.path.join(result_dir, CCDC_SEGMENTS_SUBDIRECTORY), exist_ok=True)

    os.makedirs(os.path.join(result_dir, RESIDUALS_SUBDIRECTORY), exist_ok=True)

    os.makedirs(kalman_analysis_directory(run_directory), exist_ok=True)


def generate_all_plots(run_directory, points, harmonic_flags):
    for point_index in range(len(points)):
        generate_plots(
            data=build_kalman_result_path(run_directory, point_index),
            output_path=build_kalman_analysis_path(run_directory, point_index),
            options={
                PlotType.KALMAN_VS_HARMONIC: {
                    "title": f"Kalman vs Harmonic Trend",
                    HARMONIC_TREND_LABEL: build_harmonic_trend_path(
                        run_directory, point_index
                    ),
                    HARMONIC_FLAGS_LABEL: harmonic_flags,
                },
                PlotType.KALMAN_FIT: {
                    "title": "Kalman Fit",
                },
                PlotType.KALMAN_VS_CCDC: {
                    "title": "Kalman vs CCDC",
                    CCDC.SEGMENTS.value: build_ccdc_segments_path(
                        run_directory, point_index
                    ),
                },
                PlotType.KALMAN_VS_CCDC_COEFS: {
                    "title": "Kalman vs CCDC Coefficients",
                    HARMONIC_FLAGS_LABEL: harmonic_flags,
                    CCDC.SEGMENTS.value: build_ccdc_segments_path(
                        run_directory, point_index
                    ),
                },
                PlotType.KALMAN_RETROFITTED: {
                    HARMONIC_FLAGS_LABEL: harmonic_flags,
                    Kalman.EOY_STATE.value: build_end_of_year_kalman_state_path(
                        run_directory, point_index
                    ),
                    FORWARD_TREND_LABEL: True,
                    HARMONIC_TREND_LABEL: build_harmonic_trend_path(
                        run_directory, point_index
                    ),
                },
                PlotType.KALMAN_YEARLY_FIT: {
                    HARMONIC_FLAGS_LABEL: harmonic_flags,
                    "title": "Kalman Yearly Fit",
                    Kalman.EOY_STATE.value: build_end_of_year_kalman_state_path(
                        run_directory, point_index
                    ),
                    CCDC.SEGMENTS.value: build_ccdc_segments_path(
                        run_directory, point_index
                    ),
                },
                PlotType.RESIDUALS: {
                    "title": "Residuals",
                    "residuals_path": build_residuals_path(run_directory, point_index),
                },
            },
        )


def run_continuous_kalman(
    run_directory,
    parameters,
    points,
    harmonic_flags,
    collection_parameters,
    initialization,
    include_ccdc,
):
    # create result, harmonic coefficients, and analysis directories
    setup_subdirectories(run_directory)

    # save point coordinates to file
    json.dump(points, open(build_points_path(run_directory), "w"))

    # load kalman parameters Q, R, P, X
    kalman_parameters = parameters

    threads = []
    for index, point in enumerate(points):
        thread = threading.Thread(
            target=process_point,
            args=(
                run_directory,
                kalman_parameters.copy(),
                (index, point),
                harmonic_flags,
                collection_parameters,
                initialization,
                include_ccdc,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    generate_all_plots(run_directory, points, harmonic_flags)
