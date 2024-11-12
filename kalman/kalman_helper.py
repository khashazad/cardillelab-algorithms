"""Organizes the input and calls the Kalman Filter.

The input file should contain a list of comma separated numbers on each line.
The first line should contain the values for the Q matrix (the process noise),
the second line should contain the value for R (the measurement noise), and the
third line should contain the values for the P matrix (the initial state
covariance).

Q, R, and P have a different shape depending on the number of parameters in
the state variable. For a Kalman filter whose state variable has N parameters:
Q has shape (N, N)
R has shape (1, 1)
P has shape (N, N)
Set the flags `num_sinusoid_pairs`, `include_intercept`, `include_slope` to
change the number of parameters in the state variable.

For variables that have more than one dimension they should be written row first
in the input file. e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9 will be converted to the
following matrix:
    1, 2, 3
    4, 5, 6
    7, 8, 9

The output file will be in CSV format with a single header row containing the
following columns:
    point: ID of the point (row number of the point in the given points file)
    INTP: the intercept value for the current point at this time step
    SLP: the slope value for the current point at this time step
    COS0: the cosine coefficient ""
    SIN0: the sine coefficient ""
    COS2: ""
    SIN2: ""
    COS3: ""
    SIN3: ""
State values (INPT, SLP, etc.) will be recorded to 10 significant figures in
scientific notation. Only the state values that are used in the given model are
output. The model can be adjusted using the flags `include_intercept`,
`include_slope`, and `num_sinusoid_pairs`.

The points file should specify each location where you want to apply the Kalman
filter. Each line of the points file should contain 2 numbers: the longitude of
the point and the latitude of the point followed by two strings: the date to
start running the Kalman filter and the date to stop running the kalman filter
(in YYYY-MM-dd format) all separated by a single ',' with no spaces.
"""

import argparse
import os
import json

import ee
import numpy as np
import pandas as pd
from google.api_core import retry
from pathos.pools import ProcessPool
from pprint import pprint

from kalman import kalman_filter
from lib import constants
from lib.image_collections import COLLECTIONS
from lib.utils.ee import ccdc_utils
from lib.utils import utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

NUM_MEASURES = 1  # eeek only supports one band at a time


def main(args):
    param_names = []
    if args["include_intercept"]:
        param_names.append("INTP")
    if args["include_slope"]:
        param_names.append("SLP")
    for i in range(args["num_sinusoid_pairs"]):
        param_names.extend([f"COS{i}", f"SIN{i}"])
    num_params = len(param_names)

    request_band_names = param_names.copy()
    if "store_estimate" in args and args["store_estimate"]:
        request_band_names.append("estimate")
    if "store_amplitude" in args and args["store_amplitude"]:
        request_band_names.append("amplitude")
    if "store_measurement" in args and args["store_measurement"]:
        request_band_names.append("z")
    if "store_date" in args and args["store_date"]:
        request_band_names.append("date")

    num_request_bands = len(request_band_names)

    #################################################
    ########### Read in parameters #############
    #################################################

    kalman_init = {}

    if args["input"].endswith(".json"):
        with open(args["input"], "r") as f:
            params = json.load(f)

            Q = np.array(params["Q"]).flatten()
            R = np.array(params["R"]).flatten()
            P = np.array(params["P"]).flatten()

            if (
                "initial_state" in params
                and "initialization" in args
                and args["initialization"] == "uninformative"
            ):
                kalman_init["init_image"] = np.array(params["initial_state"])
    else:
        with open(args["input"], "r") as f:
            lines = f.readlines()
        assert len(lines) == 3, "PEST parameter file must specify Q, R, P"

        Q, R, P = lines

        Q = np.array([float(x) for x in Q.split(",")])
        R = np.array([float(x) for x in R.split(",")])
        P = np.array([float(x) for x in P.split(",")])

    Q = Q.reshape(num_params, num_params)
    R = R.reshape(NUM_MEASURES, NUM_MEASURES)
    P = P.reshape(num_params, num_params)

    P = ee.Image(ee.Array(P.tolist())).rename("P")

    H = utils.sinusoidal(
        args["num_sinusoid_pairs"],
        include_slope=args["include_slope"],
        include_intercept=args["include_intercept"],
    )

    kalman_init = {
        **kalman_init,
        "F": utils.identity(num_params),
        "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        "H": H,
        "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        "num_params": num_params,
    }

    #################################################
    # Create parameters to run filter on each point #
    #################################################
    points = []
    with open(args["points"], "r") as f:
        for i, line in enumerate(f.readlines()):
            lon, lat, x1, x2, x3 = line.split(",")
            points.append(
                {
                    "index": i,
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "x0": [float(x1), float(x2), float(x3)],
                }
            )

    #################################################
    ##### Run Kalman filter across all points #######
    #################################################
    @retry.Retry()
    def process_point(kwargs):
        index = kwargs["index"]

        coords = (float(kwargs["longitude"]), float(kwargs["latitude"]))

        col = args["collection"].filterBounds(ee.Geometry.Point(coords))

        x0 = kalman_init["init_image"] if "init_image" in kalman_init else kwargs["x0"]

        x0 = ee.Image(
            ee.Array(
                np.array([float(x) for x in x0])
                .reshape(num_params, NUM_MEASURES)
                .tolist()
            )
        ).rename(constants.STATE)

        kalman_init["init_image"] = ee.Image.cat([P, x0])
        kalman_init["point_coords"] = coords

        kalman_result = kalman_filter.kalman_filter(col, **kalman_init)

        states = (
            kalman_result.map(lambda im: utils.unpack_arrays(im, param_names))
            .select(request_band_names)
            .toBands()
        )

        request = utils.build_request(coords)
        request["expression"] = states
        data = utils.compute_pixels_wrapper(request).reshape(-1, num_request_bands)

        df = pd.DataFrame(data, columns=request_band_names)
        df["point"] = [index] * df.shape[0]

        # put point as the first column
        df = df[["point"] + request_band_names]

        df = df.rename(columns={"date": "timestamp"})

        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")

        basename, ext = os.path.splitext(args["output"])

        directory = os.path.join(os.path.dirname(basename), "outputs")
        os.makedirs(directory, exist_ok=True)
        shard_path = os.path.join(directory, f"{index:03d}" + ext)
        df.to_csv(shard_path, index=False)

        return shard_path

    # with ProcessPool(nodes=40) as pool:
    #   all_output_files = pool.map(process_point, points)

    all_output_files = []
    for point in points:
        all_output_files.append(process_point(point))

    # result = process_point(points[0])
    # all_output_files = [result]

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    output_by_point = map(pd.read_csv, all_output_files)

    all_results = pd.concat(output_by_point, ignore_index=True)
    all_results.to_csv(args["output"], index=False)

    # delete intermediate files
    if "individual_outputs" not in args or not args["individual_outputs"]:
        for f in all_output_files:
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # show the module docstring in help
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="file to read parameters from",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="file to write results to",
    )
    parser.add_argument(
        "--points",
        required=True,
        help="file containing points to run kalman filter on",
    )
    parser.add_argument(
        "--collection",
        default="L8",
        help="name of image collection defined in image_collections.py",
    )
    parser.add_argument(
        "--include_intercept",
        action="store_true",
        help="if set the model will include a y intercept",
    )
    parser.add_argument(
        "--include_slope",
        action="store_true",
        help="if set the model will include a linear slope",
    )
    parser.add_argument(
        "--num_sinusoid_pairs",
        choices=[1, 2, 3],
        type=int,
        default=3,
        help="the number of sin/cos pairs to include in the model",
    )
    parser.add_argument(
        "--store_measurement",
        action="store_true",
        help="if set the measurements at each time step will be saved",
    )
    parser.add_argument(
        "--store_estimate",
        action="store_true",
        help="if set the estimate at each time step will be saved",
    )
    parser.add_argument(
        "--store_date",
        action="store_true",
        help="if set the measurement date at each time step will be saved",
    )
    parser.add_argument(
        "--store_amplitude",
        action="store_true",
        help="if set the amplitude at each time step will be saved",
    )
    parser.add_argument(
        "--include_ccdc_coefficients",
        action="store_true",
        help="if set the CCDC coefficients will be included in the output",
    )

    args = vars(parser.parse_args())
    main(args)
