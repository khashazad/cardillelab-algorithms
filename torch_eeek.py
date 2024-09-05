"""Runs the Kalman Filter for PEST parameter estimation.

The input file, containing the parameters that PEST is estimating, should
contain a list of comma separated numbers on each line. The first line should
contain the values for the Q matrix (the process noise), the second line should
contain the values for the R matrix (the measurement noise), the third line
should contain the values for the P matrix (the initial state covariance). The
fourth line should contain the values of x0 (the initial state matrix).

Q, R, P, and x0 have a different shape depending on the number of parameters in
the state variable. For a Kalman filter whose state variable has N parameters:
Q has shape (N, N)
R has shape (1, 1)
P has shape (N, N)
x0 has shape (N, 1)
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
import torch
import pandas as pd
from pathos.pools import ProcessPool
from pprint import pprint
from eeek.kalman_filter_torch import kalman_filter
import math

NUM_MEASURES = 1  # eeek only supports one band at a time

band_names = [
    "point",
    "INTP",
    "COS0",
    "SIN0",
    "estimate",
    "z",
    "date",
    # "amplitude",
]


def sinusoidal(num_sinusoid_pairs, include_slope=True, include_intercept=True):
    """Creates sinusoid function of the form a+b*t+c*cos(2pi*t)+d*sin(2pi*t)...

    Useful for H in a Kalman filter setup.

    Args:
        num_sinusoid_pairs: int, number of sine + cosine terms in the model.
        include_slope: bool, if True include a linear slope term in the model.
        include_intercept: bool, if True include a bias/intercept term in the model.

    Returns:
        function that takes a torch.Tensor and returns a torch.Tensor
    """
    num_params = 0
    if include_intercept:
        num_params += 1
    if include_slope:
        num_params += 1
    num_params += 2 * num_sinusoid_pairs

    def sinusoidal_function(t):
        """Generates sinusoidal values based on the input time t.

        Args:
            t: torch.Tensor, time variable

        Returns:
            torch.Tensor
        """
        result = torch.zeros(num_params, dtype=torch.float32)
        idx = 0
        if include_intercept:
            result[idx] = 1.0  # Intercept term
            idx += 1
        if include_slope:
            result[idx] = t  # Slope term
            idx += 1
        for i in range(num_sinusoid_pairs):
            freq = (i + 1) * 2 * math.pi
            result[idx] = torch.cos(freq * t)
            result[idx + 1] = torch.sin(freq * t)
            idx += 2
        return result

    return sinusoidal_function


def main(args, q1, q5, q9, r):
    param_names = []
    if args["include_intercept"]:
        param_names.append("INTP")
    if args["include_slope"]:
        param_names.append("SLP")
    for i in range(args["num_sinusoid_pairs"]):
        param_names.extend([f"COS{i}", f"SIN{i}"])
    num_params = len(param_names)

    #################################################
    ########### Read in PEST parameters #############
    #################################################
    parameters = {}

    Q = torch.tensor(
        [[q1, 0.0, 0.0], [0.0, q5, 0.0], [0.0, 0.0, q9]], dtype=torch.float32
    )
    parameters["process noise (Q)"] = Q.tolist()

    R = torch.tensor([r], dtype=torch.float32)
    parameters["measurement noise (R)"] = R.tolist()

    P = torch.tensor([[0.00101, 0.0, 0.0], [0.0, 0.00222, 0.0], [0.0, 0.0, 0.00333]])
    parameters["initial state covariance matrix (P)"] = P.tolist()

    H = sinusoidal(
        args["num_sinusoid_pairs"],
        include_slope=args["include_slope"],
        include_intercept=args["include_intercept"],
    )

    kalman_init = {
        "F": torch.eye(num_params),
        "Q": Q,
        "H": H,
        "R": R,
        "num_params": num_params,
    }

    with open(args["parameters_output"], "w") as f:
        json.dump(parameters, f, indent=4)

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

    all_measurements = pd.read_csv(args["measurements"])

    #################################################
    ##### Run Kalman filter across all points #######
    #################################################
    def process_point(kwargs):
        index = kwargs["index"]

        measurements_for_point = all_measurements[
            all_measurements["point"] == kwargs["index"]
        ]
        measurements = measurements_for_point[["swir", "date"]].values.tolist()

        x0 = torch.tensor(kwargs["x0"], dtype=torch.float32).reshape(
            num_params, NUM_MEASURES
        )

        kalman_result = kalman_filter(measurements, x0, P, **kalman_init)

        output = [
            [
                index,
                row[0][0].item(),
                row[0][1].item(),
                row[0][2].item(),
                row[1].item(),
                row[2],
                row[3],
            ]
            for row in kalman_result
        ]

        df = pd.DataFrame(output, columns=band_names)

        df.sort_values(by=["point", "date"], inplace=True)

        # df = df[["point"] + band_names]

        basename, ext = os.path.splitext(args["output"])
        shard_path = basename + f"-{index:06d}" + ext
        df.to_csv(shard_path, index=False)

        return shard_path

    with ProcessPool(nodes=40) as pool:
        all_output_files = pool.map(process_point, points)

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    all_results = pd.concat(map(pd.read_csv, all_output_files), ignore_index=True)
    all_results.to_csv(args["output"], index=False)

    # delete intermediate files
    for f in all_output_files:
        os.remove(f)

    return all_results[["INTP", "COS0", "SIN0"]].values.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # show the module docstring in help
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--measurements",
        required=True,
        help="file containing measurements",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="file to write results to",
    )
    parser.add_argument(
        "--parameters_output",
        required=True,
        help="file to write parameters to",
    )
    parser.add_argument(
        "--points",
        required=True,
        help="file containing points to run kalman filter on",
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
    q1 = 0.00125
    q5 = 0.000125
    q9 = 0.000125
    r = 0.003
    main(args, q1, q5, q9, r)
