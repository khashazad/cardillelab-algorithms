"""
Test our earth engine implementation against a well established local python
implementation to ensure we implemented the kalman filter properly.
"""
import math
import string
import itertools
from multiprocessing import Pool, Manager

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import ee
from filterpy.kalman import ExtendedKalmanFilter as EKF

from eeek.kalman_filter import kalman_filter
from eeek import utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


INITIALIZATIONS = {
    "F": np.eye(3),
    "Q": np.diag([0.001, 0.0005, 0.00025]),
    "R": np.array([[0.1234]]),
    "x": np.array([[0.04, 0.04, 0.05]]).T,
    "P": np.diag([0.04, 0.04, 0.05]),
    "num_params": 3,
    "num_measures": 1,
}


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
        "fileFormat": "NUMPY_NDARRAY",
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


def create_initializations(num_params, num_measures, x, P, F, Q, R):
    """For given inputs create matching initializations for local and ee EKF.

    Create the measurement functions as a + bcos(t) + csin(t) + ... based on
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
        t *= 2 * math.pi
        H = [1]
        for _ in range((num_params - 1) // 2):
            H.extend([np.cos(t), np.sin(t)])
        return np.array([H])

    def Hx(x, t):
        return Hj(x, t) @ x

    local_init = {"ekf": ekf, "Hj": Hj, "Hx": Hx}

    # build H for ee version
    def H_fn(t, **kwargs):
        t = t.multiply(2 * math.pi)
        images = [ee.Image.constant(1.0)]
        for _ in range((num_params - 1) // 2):
            images.extend([t.cos(), t.sin()])
        H = ee.Image.cat(*images).toArray(0)
        return H.arrayReshape(ee.Image(ee.Array([1, -1])), 2)

    ee_init = {
        "init_image": ee.Image.cat(
            ee.Image(ee.Array(x.tolist())).rename("x"),
            ee.Image(ee.Array(P.tolist())).rename("P"),
        ),
        "F": lambda **kwargs: ee.Image(ee.Array(F.tolist())),
        "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        "H": H_fn,
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


def compare_at_point(index, point, output_list, max_images=30, **init_args):
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
    kalman_input = np.squeeze(
        structured_to_unstructured(ee.data.computePixels(request))
    )

    ee_result = kalman_filter(col, **ee_init)
    ee_result = ee_result.map(
        lambda im: utils.unpack_arrays(im, INITIALIZATIONS["num_params"])
    )

    # get the ee kalman states as local data
    param_names = list(string.ascii_lowercase)[: init_args["num_params"]]
    request["expression"] = ee_result.select(param_names).toBands()
    ee_states = np.squeeze(
        structured_to_unstructured(ee.data.computePixels(request))
    ).reshape((-1, init_args["num_params"], init_args["num_measures"]))

    # get the ee kalman state covariances as local data
    cov_names = [f"cov_{x}_{y}" for x in param_names for y in param_names]
    request["expression"] = ee_result.select(cov_names).toBands()
    ee_covariances = np.squeeze(
        structured_to_unstructured(ee.data.computePixels(request))
    ).reshape((-1, init_args["num_params"], init_args["num_params"]))

    # get the local kalman state and state covariances
    local_states, local_covariances = run_local_kalman(
        kalman_input, image_dates, **local_init
    )

    output_list[index] = np.allclose(local_states, ee_states) and np.allclose(
        local_covariances, ee_covariances
    )


def _multiprocessing_fn(args):
    """Wrapper for compare_at_point to be passed to multiprocessing.Pool.map

    Must be defined at the top level to be pickleable (which is necessary for
    multiprocessing)

    TODO: using pool.starmap possibly simplifies args

    Args:
        args: (int, ((int, int), managed list)), this gross input type is
        necessary to allow this function to be defined at the top scope (and
        therefore be pickleable) while still being able to pass in a managed
        list from inside the scope of a context manager.

    Returns:
        None
    """
    kwargs = {
        "index": args[0],
        "point": args[1][0],
        "output_list": args[1][1],
        **INITIALIZATIONS,
    }
    compare_at_point(**kwargs)


def test_correctness(N=100):
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

    with Manager() as manager:
        test_results = manager.list([None] * N)
        with Pool(40) as pool:
            pool.map(
                _multiprocessing_fn,
                enumerate(zip(points, itertools.repeat(test_results))),
            )

        print(test_results)
        assert np.sum(test_results) == N
