import os
import json
from lib.constants import (
    Index,
    Initialization,
    Sensor,
)
from lib.study_packages import (
    get_tag,
    get_points,
)
from lib.constants import Harmonic
from lib.paths import (
    build_kalman_run_directory,
)
from lib.constants import KalmanModels
from kalman.kalman_caller_helper import run_continuous_kalman

# could be omitted
RUN_ID = ""

# whether to include the ccdc coefficients in the output
INCLUDE_CCDC_COEFFICIENTS = True

# initialization method
INITIALIZATION = Initialization.POSTHOC

# Parameters
COLLECTION_PARAMETERS = {
    "index": Index.SWIR,
    "sensors": [Sensor.L8, Sensor.L9],
    "years": [2017, 2018, 2019],
    "point_group": "pnw_7",
    "day_step_size": 6,
    "start_doy": 1,
    "end_doy": 365,
    "cloud_cover_threshold": 20,
    "initialization": INITIALIZATION,
}

TAG = get_tag(**COLLECTION_PARAMETERS)
POINTS = get_points(COLLECTION_PARAMETERS["point_group"])

# Get the directory of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Define the run directory based on the current timestamp
run_directory = build_kalman_run_directory(script_directory, TAG, None, RUN_ID)

model_parameters = {
    KalmanModels.UNIMODAL.value: {
        "Q": [0.00125, 0.000125, 0.000125],
        "R": [0.003],
        "P": [0.04, 0.04, 0.03],
    },
    KalmanModels.UNIMODAL_WITH_SLOPE.value: {
        "Q": [
            0.00125,
            0.000125,
            0.000125,
            0.000125,
        ],
        "R": [0.02345],
        "P": [0.0101, 0.0111, 0.0222, 0.0333],
    },
    KalmanModels.BIMODAL.value: {
        "Q": [
            0.00125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
        ],
        "R": [0.02345],
        "P": [0.0101, 0.0222, 0.0333, 0.0444, 0.0555],
    },
    KalmanModels.BIMODAL_WITH_SLOPE.value: {
        "Q": [
            0.00125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
        ],
        "R": [0.02345],
        "P": [0.0101, 0.0111, 0.0222, 0.0333, 0.0444, 0.0555],
    },
    KalmanModels.TRIMODAL.value: {
        "Q": [
            0.00125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
        ],
        "R": [0.02345],
        "P": [0.0101, 0.0222, 0.0333, 0.0444, 0.0555, 0.0666, 0.0777],
    },
    KalmanModels.TRIMODAL_WITH_SLOPE.value: {
        "Q": [
            0.00125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
            0.000125,
        ],
        "R": [0.02345],
        "P": [0.0101, 0.0111, 0.0222, 0.0333, 0.0444, 0.0555, 0.0666, 0.0777],
    },
}

model_harmonic_flags = {
    KalmanModels.UNIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.UNIMODAL.value: True,
    },
    KalmanModels.UNIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.UNIMODAL.value: True,
    },
    KalmanModels.BIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.BIMODAL.value: True,
    },
    KalmanModels.BIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.BIMODAL.value: True,
    },
    KalmanModels.TRIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.TRIMODAL.value: True,
    },
    KalmanModels.TRIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.TRIMODAL.value: True,
    },
}

models = [
    KalmanModels.UNIMODAL.value,
    # KalmanModels.UNIMODAL_WITH_SLOPE.value,
    # KalmanModels.BIMODAL.value,
    # KalmanModels.BIMODAL_WITH_SLOPE.value,
    # KalmanModels.TRIMODAL.value,
    KalmanModels.TRIMODAL_WITH_SLOPE.value,
]

if __name__ == "__main__":

    # Ensure the run directory exists
    os.makedirs(run_directory, exist_ok=True)

    for model in models:
        sub_run_directory = os.path.join(run_directory, model)
        os.makedirs(sub_run_directory, exist_ok=True)

        parameters = model_parameters.get(model, {})

        with open(os.path.join(sub_run_directory, "parameters.json"), "w") as f:
            json.dump(parameters, f)

        print(f"Running model: {model}")

        run_continuous_kalman(
            sub_run_directory,
            model_parameters.get(model, {}),
            POINTS,
            model_harmonic_flags.get(model, {}),
            COLLECTION_PARAMETERS,
            INITIALIZATION,
            INCLUDE_CCDC_COEFFICIENTS,
        )
