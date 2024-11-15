import ee
import json
import numpy as np
from lib import constants
from lib.utils import utils
from lib.constants import (
    Harmonic,
    Recording,
    Kalman,
    ESTIMATE,
    AMPLITUDE,
    TIMESTAMP,
)
from lib.constants import NUM_MEASURES


def read_parameters_from_file(parameters_file_path):
    with open(parameters_file_path, "r") as file:
        parameters = json.load(file)

        return {
            Kalman.Q: parameters.get("Q", []),
            Kalman.R: parameters.get("R", []),
            Kalman.P: parameters.get("P", []),
            Kalman.X: parameters.get("X", []),
        }


def get_harmonic_params(harmonic_params):
    params = []

    if harmonic_params.get(Harmonic.INTERCEPT.value):
        params.append(Harmonic.INTERCEPT.value)
    if harmonic_params.get(Harmonic.SLOPE.value):
        params.append(Harmonic.SLOPE.value)
    for i in range(harmonic_params.get(Harmonic.MODALITY.value, 1)):
        params.extend(
            [
                f"{Harmonic.COS.value}{i}",
                f"{Harmonic.SIN.value}{i}",
            ]
        )

    return params


def parse_parameters_and_bands(harmonic_params, flags):
    param_names = get_harmonic_params(harmonic_params)

    band_names = param_names.copy()

    if flags.get(Recording.ESTIMATE, False):
        band_names.append(ESTIMATE)
    if flags.get(Recording.AMPLITUDE, False):
        band_names.append(AMPLITUDE)
    if flags.get(Recording.TIMESTAMP, False):
        band_names.append(TIMESTAMP)

    band_names.append(Kalman.Z.value)

    return param_names, band_names


def setup_kalman_init(kalman_parameters, harmonic_params):
    harmonic_params_names = get_harmonic_params(harmonic_params)
    num_params = len(harmonic_params_names)

    Q = np.array(kalman_parameters.get("Q", [])).flatten()
    R = np.array(kalman_parameters.get("R", [])).flatten()
    P = np.array(kalman_parameters.get("P_0", [])).flatten()
    X = np.array(kalman_parameters.get("X_0", [])).flatten()

    assert (
        len(Q) == num_params * num_params
    ), f"Q must be a square matrix of size {num_params}x{num_params}"
    assert len(R) == NUM_MEASURES, f"R must be a vector of size {NUM_MEASURES}"
    assert (
        len(P) == num_params * num_params
    ), f"P_0 must be a square matrix of size {num_params}x{num_params}"
    assert len(X) == num_params, f"X_0 must be a vector of size {num_params}"

    H = utils.sinusoidal(
        harmonic_params.get(Harmonic.MODALITY.value, 1),
        include_slope=harmonic_params.get(Harmonic.SLOPE.value, False),
        include_intercept=harmonic_params.get(Harmonic.INTERCEPT.value, False),
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


def unpack_kalman_results(image, param_names):
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
    z = image.select(Kalman.Z.value).arrayProject([0]).arrayFlatten([[Kalman.Z.value]])
    x = image.select(Kalman.X.value).arrayProject([0]).arrayFlatten([param_names])
    P = image.select(Kalman.P.value).arrayFlatten(
        [[Kalman.COV_PREFIX.value + x for x in param_names], param_names]
    )
    estimate = image.select(ESTIMATE).arrayProject([0]).arrayFlatten([[ESTIMATE]])

    return image.addBands(ee.Image.cat(z, x, P, estimate), overwrite=True)
