"""
Test our earth engine implementation against a well established local python
implementation to ensure we implemented the kalman filter properly.
"""
import io
import math
import string
import itertools
from multiprocessing import Pool, Manager

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import ee
import pytest
from filterpy.kalman import ExtendedKalmanFilter as EKF

from eeek.kalman_filter import kalman_filter
from eeek import utils, constants

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

TEST_PARAMS = itertools.product((50,), (3, 5, 7), (1,), (True, False))


def make_random_init(num_params, num_measures, linear_term, seed):
    rng = np.random.default_rng(int(abs(seed)))

    return {
        "F": np.eye(num_params),
        "Q": np.diag(rng.uniform(size=num_params)),
        "R": rng.uniform(size=(num_measures, num_measures)),
        "x": rng.uniform(size=(num_params, 1)),
        "P": np.diag(rng.uniform(size=num_params)),
        "num_params": num_params,
        "num_measures": num_measures,
        "linear_term": linear_term,
    }


def compute_pixels_wrapper(request):
    """Wraps ee.data.computePixels to allow loading larger npy files.

    Expects request to have fileFormat == "NPY"

    Args:
        request: dict, passed to ee.data.computePixels

    Returns:
        np.ndarray
    """
    result = ee.data.computePixels(request)
    return np.squeeze(
        structured_to_unstructured(np.load(io.BytesIO(result), allow_pickle=True))
    )


def get_utm_from_lonlat(lon, lat):
    """Get the EPSG CRS Code for the UTM zone of a given lon, lat pait.

    Based on: https://stackoverflow.com/a/9188972

    Args:
        lon: float
        lat: float

    Returns:
        string
    """
    offset = 32601 if lat >= 0 else 32701
    return "EPSG:" + str(offset + (math.floor((lon + 180) / 6) % 60))


def build_request(point, scale=10):
    """Create a 1x1 numpy ndarray computePixels request at the given point.

    Args:
        point: (float, float), lat lon coordinates
        scale: int

    Returns:
        dict, passable to ee.data.computePixels, caller must set 'expression'
    """
    crs = get_utm_from_lonlat(*point)
    proj = ee.Projection(crs)
    geom = ee.Geometry.Point(point)
    coords = ee.Feature(geom).geometry(1, proj).getInfo()["coordinates"]
    request = {
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {
                "width": 1,
                "height": 1,
            },
            "affineTransform": {
                "scaleX": scale,
                "shearX": 0,
                "translateX": coords[1],
                "shearY": 0,
                "scaleY": -scale,
                "translateY": coords[1],
            },
            "crsCode": crs,
        },
    }
    return request


def create_initializations(num_params, num_measures, linear_term, x, P, F, Q, R):
    """For given inputs create matching initializations for local and ee EKF.

    Create the measurement functions as: a + b*cos(t) + c*sin(t) + ... based on
    num_params and num_measures

    Args:
        num_params: int, number of variables in state per measurement
        num_measures: int, number of measurements/bands
        x: np.ndarray, initial state variable
        P: np.ndarray, initial state covariance
        F: np.ndarray, state process function
        Q: np.ndarray, state process noise
        R: np.ndarray, measurement noise

    Returns:
        (dict, dict) the local initializations and ee initializations
        respectively.
    """
    ekf = EKF(num_params, num_measures)
    ekf.x = x.copy()
    ekf.P = P.copy()
    ekf.F = F.copy()
    ekf.Q = Q.copy()
    ekf.R = R.copy()

    # build H for local version
    def Hj(x, t):
        H = [1]
        if linear_term:
            H.append(t)
        t *= 2 * math.pi
        for _ in range((num_params - 1) // 2):
            H.extend([np.cos(t), np.sin(t)])
        return np.array([H])

    def Hx(x, t):
        return Hj(x, t) @ x

    local_init = {"ekf": ekf, "Hj": Hj, "Hx": Hx}

    ee_init = {
        "init_image": ee.Image.cat(
            ee.Image(ee.Array(x.tolist())).rename(constants.STATE),
            ee.Image(ee.Array(P.tolist())).rename(constants.COV),
        ),
        "F": lambda **kwargs: ee.Image(ee.Array(F.tolist())),
        "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        "H": utils.sinusoidal(num_params, linear_term),
        "num_params": num_params,
    }

    return local_init, ee_init


def run_local_kalman(inputs, times, ekf, Hj, Hx):
    """Run a local kalman filter defined by ekf, Hj, and Hx.

    Args:
        inputs: np.ndarray, the input measurements
        times: np.ndarray, the time of measurement (passed to Hj and Hx)
        ekf: filterpy.kalman.ExtendedKalmanFilter, initilized kalman filter
        Hj: function, computes the Jacobian of the H matrix
        Hx: function, computes the H matrix for given state

    Returns:
        (np.ndarray, np.ndarray), the states and covariances from each step of
        running the kalman filter.
    """
    states = []
    covariances = []
    for val, t in zip(inputs, times):
        ekf.predict_update(val, Hj, Hx, (t,), (t,))
        states.append(ekf.x)
        covariances.append(ekf.P)
    return np.array(states), np.array(covariances)


def compare_at_point(
    index, point, output_list, max_images, num_params, num_measures, linear_term
):
    """Compares outputs of a local and ee kalman filter on the given point.

    Stores the comparison result at output_list[index]

    The comparison is made using np.allclose, the state (x) and covariance
    matrix (P) from each step of running the kalman filters are compared.

    Args:
        index: int,
        point: (int, int): lon, lat coordinates
        output_list: multiprocessing managed list
        init_kwargs: passed to create_initializations
        max_images: max images to run kalman filter for at the point.

    Returns:
        None
    """
    if linear_term:
        num_params += 1

    init_args = make_random_init(num_params, num_measures, linear_term, point[0])
    local_init, ee_init = create_initializations(**init_args)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee.Geometry.Point(point))
        .limit(max_images)
        .select("B12")
    )

    # get date of each image as local data
    date_collection = col.map(
        lambda im: ee.Feature(
            None,
            {"time": im.date().difference("2016-01-01", "year")},
        )
    )
    image_dates = np.array(
        ee.data.computeFeatures(
            {
                "expression": date_collection,
                "fileFormat": "PANDAS_DATAFRAME",
            }
        )["time"]
    )

    # get the raw collection as local data
    request = build_request(point)
    request["expression"] = col.toBands()
    kalman_input = compute_pixels_wrapper(request)


    param_names = list(string.ascii_lowercase)[:num_params]

    ee_result = kalman_filter(col, **ee_init)
    ee_result = ee_result.map(lambda im: utils.unpack_arrays(im, param_names))

    # get the ee kalman states as local data
    request["expression"] = ee_result.select(param_names).toBands()
    state_shape = (-1, num_params, num_measures)
    ee_states = compute_pixels_wrapper(request).reshape(state_shape)

    # get the ee kalman state covariances as local data
    cov_names = [f"cov_{x}_{y}" for x in param_names for y in param_names]
    request["expression"] = ee_result.select(cov_names).toBands()
    cov_shape = (-1, num_params, num_params)
    ee_covariances = compute_pixels_wrapper(request).reshape(cov_shape)

    # get the local kalman state and state covariances
    local_states, local_covariances = run_local_kalman(
        kalman_input, image_dates, **local_init
    )

    tol = 1e-8
    output_list[index] = np.allclose(
        local_states, ee_states, rtol=tol, atol=tol
    ) and np.allclose(local_covariances, ee_covariances, rtol=tol, atol=tol)


@pytest.mark.parametrize("N,num_params,num_measures,linear_term", TEST_PARAMS)
def test_correctness(N, num_params, num_measures, linear_term):
    # create a sample of N random points over North America
    roi = ee.Geometry.Rectangle([(-116, 33), (-82, 54)])
    samples = ee.Image.constant(1).sample(
        region=roi,
        scale=10,
        numPixels=N,
        seed=42,
        geometries=True,
    )
    points = samples.geometry().coordinates().getInfo()

    # use the largest number of images while avoiding computePixels' band limit
    max_images = 1024 // ((num_params + int(linear_term)) ** 2)

    with Manager() as manager:
        test_results = manager.list([None] * N)
        with Pool(40) as pool:
            pool.starmap(
                compare_at_point,
                zip(
                    range(len(points)),
                    points,
                    itertools.repeat(test_results),
                    itertools.repeat(max_images),
                    itertools.repeat(num_params),
                    itertools.repeat(num_measures),
                    itertools.repeat(linear_term),
                ),
            )

        assert np.sum(test_results) == N
