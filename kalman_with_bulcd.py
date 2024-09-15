import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

import ee
import numpy as np
import pandas as pd
from google.api_core import retry
from pathos.pools import ProcessPool
from pprint import pprint

from kalman import kalman_filter
from lib.image_collections import COLLECTIONS
from utils.filesystem import write_json
from utils import utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_MEASURES = 1  # eeek only supports one band at a time

kalman_params = {
    "Q": [0.00125, 0.000125, 0.000125],
    "R": [0.003],
    "P": [0.00101, 0.00222, 0.00333],
    "change_probability_threshold": 0.65,
    "Q_scale_factor": 10.0,
}


def parse_points(points_file):
    points = []
    with open(points_file, "r") as f:
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
    return points


def main(args):
    param_names = []
    if args["include_intercept"]:
        param_names.append("INTP")
    if args["include_slope"]:
        param_names.append("SLP")
    for i in range(args["num_sinusoid_pairs"]):
        param_names.extend([f"COS{i}", f"SIN{i}"])
    num_params = len(param_names)

    request_band_names = [*param_names.copy(), "estimate", "amplitude", "z", "date"]
    num_request_bands = len(request_band_names)

    Q, R, P = kalman_params["Q"], kalman_params["R"], kalman_params["P"]

    Q = np.array([float(x) for x in Q]).reshape(num_params, num_params)
    R = np.array([float(x) for x in R.split(",")]).reshape(NUM_MEASURES, NUM_MEASURES)
    P_arr = np.array([float(x) for x in P.split(",")]).reshape(num_params, num_params)

    P = ee.Image(ee.Array(P_arr.tolist())).rename("P")

    H = utils.sinusoidal(
        args["num_sinusoid_pairs"],
        include_slope=args["include_slope"],
        include_intercept=args["include_intercept"],
    )

    kalman_init = {
        "F": utils.identity(num_params),
        "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        "H": H,
        "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        "num_params": num_params,
    }

    write_json(
        os.path.join(SCRIPT_DIR, "eeek_params.json"),
        {
            "process_noise": Q.tolist(),
            "measurement_noise": R.tolist(),
            "initial_state_covariance": P_arr.tolist(),
        },
    )

    points = parse_points(args["points"])

    ###########################################################
    ##### Run Kalman filter and bulcd across all points #######
    ###########################################################
    @retry.Retry()
    def process_point(kwargs):
        index = kwargs["index"]

        coords = (float(kwargs["longitude"]), float(kwargs["latitude"]))

        col = COLLECTIONS[args["collection"]].filterBounds(ee.Geometry.Point(coords))

        x0 = ee.Image(
            ee.Array(np.array(kwargs["x0"]).reshape(num_params, NUM_MEASURES).tolist())
        ).rename("x")

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

        basename, ext = os.path.splitext(args["output"])
        shard_path = basename + f"-{index:06d}" + ext
        df.to_csv(shard_path, index=False)

        return shard_path

    with ProcessPool(nodes=40) as pool:
        all_output_files = pool.map(process_point, points)

    # result = process_point(points[0])
    # all_output_files = [result]

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    all_results = pd.concat(map(pd.read_csv, all_output_files), ignore_index=True)
    all_results.to_csv(args["output"], index=False)

    # last_row = all_results.iloc[-1]
    # new_df = pd.DataFrame([last_row])
    # new_df.to_csv(args.output, index=False)

    # delete intermediate files
    for f in all_output_files:
        os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # show the module docstring in help
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    args = vars(parser.parse_args())
    main(args)
