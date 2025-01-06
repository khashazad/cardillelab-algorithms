from pprint import pprint
import ee
import json
import numpy as np
from lib import constants
from lib.utils import utils
from lib.constants import (
    CCDC,
    ESTIMATE_PREDICTED_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_TAGS,
    Harmonic,
    KalmanRecordingFlags,
    Kalman,
    ESTIMATE_LABEL,
    TIMESTAMP_LABEL,
)
from lib.constants import NUM_MEASURES
from lib.utils.harmonic import parse_harmonic_params


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
        band_names.append(ESTIMATE_LABEL)

    if recording_flags.get(KalmanRecordingFlags.ESTIMATE_PREDICTED, False):
        band_names.append(ESTIMATE_PREDICTED_LABEL)

    # measurement
    if recording_flags.get(KalmanRecordingFlags.MEASUREMENT, False):
        band_names.append(Kalman.Z.value)

    # ccdc coefficients
    if recording_flags.get(KalmanRecordingFlags.CCDC_COEFFICIENTS, False):
        band_names.extend(
            [*[f"{CCDC.BAND_PREFIX.value}_{x}" for x in HARMONIC_TAGS], CCDC.FIT.value]
        )

    # timestamp
    if recording_flags.get(KalmanRecordingFlags.TIMESTAMP, False):
        band_names.append(TIMESTAMP_LABEL)

    if recording_flags.get(KalmanRecordingFlags.FRACTION_OF_YEAR, False):
        band_names.append(FRACTION_OF_YEAR_LABEL)

    return band_names


def setup_kalman_init(kalman_parameters, harmonic_flags):

    harmonic_params, NUM_SINUSOID_PAIRS = parse_harmonic_params(harmonic_flags)

    num_params = len(harmonic_params)

    R = np.array(kalman_parameters.get(Kalman.R.value, [])).flatten()
    X = np.array(kalman_parameters.get(Kalman.X.value, [])).flatten()

    Q = np.array(kalman_parameters.get(Kalman.Q.value, []))
    if len(Q.shape) == 1:
        Q = np.diag(Q).flatten()

    P = np.array(kalman_parameters.get(Kalman.P.value, []))
    if len(P.shape) == 1:
        P = np.diag(P).flatten()

    Q = Q.flatten()
    P = P.flatten()

    assert (
        len(Q) == num_params**2
    ), f"Q must be a square matrix of size {num_params}x{num_params} or an array of size {num_params**2}, got {Q.shape}"
    assert (
        len(R) == NUM_MEASURES
    ), f"R must be a vector of size {NUM_MEASURES}, got {R.shape}"
    assert (
        len(P) == num_params**2
    ), f"P must be a square matrix of size {num_params} x {num_params} or an array of size {num_params**2}, got {P.shape}"

    assert (
        len(X) == num_params
    ), f"X must be a vector of size {num_params}, got {X.shape}"

    H = utils.sinusoidal(
        NUM_SINUSOID_PAIRS,
        include_slope=harmonic_flags.get(Harmonic.SLOPE.value, False),
        include_intercept=harmonic_flags.get(Harmonic.INTERCEPT.value, False),
    )

    Q = np.array([float(x) for x in Q]).reshape(num_params, num_params)
    R = np.array([float(x) for x in R]).reshape(NUM_MEASURES, NUM_MEASURES)

    P = np.array([float(x) for x in P]).reshape(num_params, num_params)
    X = np.array([float(x) for x in X]).reshape(num_params, NUM_MEASURES)

    P = ee.Image(ee.Array(P.tolist())).rename(Kalman.P.value)
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
        estimate = (
            image.select(ESTIMATE_LABEL)
            .arrayProject([0])
            .arrayFlatten([[ESTIMATE_LABEL]])
        )
        bands.append(estimate)

    if recording_flags.get(KalmanRecordingFlags.ESTIMATE_PREDICTED, False):
        estimate_predicted = (
            image.select(ESTIMATE_PREDICTED_LABEL)
            .arrayProject([0])
            .arrayFlatten([[ESTIMATE_PREDICTED_LABEL]])
        )
        bands.append(estimate_predicted)

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

    return image.addBands(ee.Image.cat(bands), overwrite=True)
