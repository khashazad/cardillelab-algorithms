import ee
import json
import numpy as np
from lib import constants
from lib.utils import utils
from lib.constants import (
    FRACTION_OF_YEAR,
    Harmonic,
    KalmanRecordingFlags,
    Kalman,
    ESTIMATE,
    TIMESTAMP,
)
from lib.constants import NUM_MEASURES


def read_parameters_from_file(parameters_file_path):
    with open(parameters_file_path, "r") as file:
        parameters = json.load(file)

        return {
            Kalman.Q: parameters.get(Kalman.Q.value, []),
            Kalman.R: parameters.get(Kalman.R.value, []),
            Kalman.P: parameters.get(Kalman.P.value, []),
            Kalman.X: parameters.get(Kalman.X.value, []),
        }


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

    for i in range(NUM_SINUSOID_PAIRS):
        param_names.extend(
            [
                f"{Harmonic.COS.value}{i}",
                f"{Harmonic.SIN.value}{i}",
            ]
        )

    return param_names, NUM_SINUSOID_PAIRS


def parse_band_names(recording_flags, harmonic_flags):
    band_names = []

    harmonic_params, _ = parse_harmonic_params(harmonic_flags)

    # state
    if recording_flags.get(KalmanRecordingFlags.STATE, False):
        band_names.extend(harmonic_params)

    # state covariance
    if recording_flags.get(KalmanRecordingFlags.STATE_COV, False):
        band_names.extend([f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params])

    # estimate
    if recording_flags.get(KalmanRecordingFlags.ESTIMATE, False):
        band_names.append(ESTIMATE)

    # measurement
    if recording_flags.get(KalmanRecordingFlags.MEASUREMENT, False):
        band_names.append(Kalman.Z.value)

    # ccdc coefficients
    if recording_flags.get(KalmanRecordingFlags.CCDC_COEFFICIENTS, False):
        band_names.extend(
            [
                f"{Harmonic.COS.value}{i}",
                f"{Harmonic.SIN.value}{i}",
            ]
            for i in range(3)
        )

    # timestamp
    if recording_flags.get(KalmanRecordingFlags.TIMESTAMP, False):
        band_names.append(TIMESTAMP)

    if recording_flags.get(KalmanRecordingFlags.FRACTION_OF_YEAR, False):
        band_names.append(FRACTION_OF_YEAR)

    return band_names


def setup_kalman_init(kalman_parameters, harmonic_flags):

    harmonic_params, NUM_SINUSOID_PAIRS = parse_harmonic_params(harmonic_flags)

    num_params = len(harmonic_params)

    Q = np.array(kalman_parameters.get(Kalman.Q.value, [])).flatten()
    R = np.array(kalman_parameters.get(Kalman.R.value, [])).flatten()
    P = np.array(kalman_parameters.get(Kalman.P.value, [])).flatten()
    X = np.array(kalman_parameters.get(Kalman.X.value, [])).flatten()

    assert (
        len(Q) == num_params * num_params
    ), f"Q must be a square matrix of size {num_params}x{num_params}"
    assert len(R) == NUM_MEASURES, f"R must be a vector of size {NUM_MEASURES}"
    assert (
        len(P) == num_params**2
    ), f"P must be a square matrix of size {num_params} x {num_params}"
    assert len(X) == num_params, f"X must be a vector of size {num_params}"

    H = utils.sinusoidal(
        NUM_SINUSOID_PAIRS,
        include_slope=harmonic_flags.get(Harmonic.SLOPE.value, False),
        include_intercept=harmonic_flags.get(Harmonic.INTERCEPT.value, False),
    )

    Q = np.array([float(x) for x in Q]).reshape(num_params, num_params)
    R = np.array([float(x) for x in R]).reshape(NUM_MEASURES, NUM_MEASURES)

    P = np.array([float(x) for x in P]).reshape(num_params, num_params)
    P = ee.Image(ee.Array(P.tolist())).rename(Kalman.P.value)

    X = np.array([float(x) for x in X]).reshape(num_params, NUM_MEASURES)
    X = ee.Image(ee.Array(X.tolist())).rename(Kalman.X.value)

    return {
        Kalman.F.value: utils.identity(num_params),
        Kalman.Q.value: lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        Kalman.H.value: H,
        Kalman.R.value: lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        Kalman.INITIAL_STATE.value: ee.Image.cat([P, X]),
    }


def unpack_kalman_results(
    image,
    harmonic_params,
    recording_flags=[KalmanRecordingFlags.ESTIMATE, KalmanRecordingFlags.MEASUREMENT],
):
    """Unpack array image into separate bands.

    Can be mapped across the output of kalman_filter.

    Args:
        image: ee.Image
        param_names: list[str], the names to give each parameter in the state

    Returns:
        ee.Image with bands for each state variable, and the covariance between
        each pair of state variables and all bands from the original input
        image.
    """

    bands = []

    if recording_flags.get(KalmanRecordingFlags.MEASUREMENT, False):
        z = (
            image.select(Kalman.Z.value)
            .arrayProject([0])
            .arrayFlatten([[Kalman.Z.value]])
        )
        bands.append(z)

    if recording_flags.get(KalmanRecordingFlags.STATE, False):
        x = (
            image.select(Kalman.X.value)
            .arrayProject([0])
            .arrayFlatten([harmonic_params])
        )
        bands.append(x)

    if recording_flags.get(KalmanRecordingFlags.ESTIMATE, False):
        estimate = image.select(ESTIMATE).arrayProject([0]).arrayFlatten([[ESTIMATE]])
        bands.append(estimate)

    if recording_flags.get(KalmanRecordingFlags.TIMESTAMP, False):
        bands.append(image.select(TIMESTAMP))

    if recording_flags.get(KalmanRecordingFlags.STATE_COV, False):
        P = (
            image.select(Kalman.P.value)
            .arrayFlatten(
                [
                    [f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params],
                    harmonic_params,
                ]
            )
            .select(
                [f"{Kalman.COV_PREFIX.value}_{x}_{x}" for x in harmonic_params],
                [f"{Kalman.COV_PREFIX.value}_{x}" for x in harmonic_params],
            )
        )
        bands.append(P)

    if recording_flags.get(KalmanRecordingFlags.FRACTION_OF_YEAR, False):
        bands.append(image.select(FRACTION_OF_YEAR))

    return ee.Image.cat(bands).copyProperties(image)
