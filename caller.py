import os

import ee
import numpy as np
import pandas as pd
from google.api_core import retry
from pathos.pools import ProcessPool
from pprint import pprint

from kalman_with_bulcd.kalman_with_bulcd import kalman_with_bulcd
from utils.filesystem import write_json
from utils import utils
from kalman_with_bulcd.parameters import run_specific_parameters
from kalman_with_bulcd.organize_inputs import organize_inputs

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "kalman_with_bulcd", "run")
POINTS_FILE = os.path.join(RUN_DIR, "points.csv")

NUM_MEASURES = 1  # eeek only supports one band at a time


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


def get_param_names(modality):
    param_names = []
    if modality["constant"]:
        param_names.append("INTP")
    if modality["linear"]:
        param_names.append("SLP")
    if modality["unimodal"]:
        param_names.append("COS0")
        param_names.append("SIN0")
    if modality["bimodal"]:
        param_names.append("COS1")
        param_names.append("SIN1")
    if modality["trimodal"]:
        param_names.append("COS2")
        param_names.append("SIN2")
    return param_names, len(param_names)


def main():
    parameters = run_specific_parameters()
    modality = parameters["modality_dictionary"]
    kalman_params = parameters["kalman_params"]

    param_names, num_params = get_param_names(modality)

    request_band_names = [*param_names.copy(), "estimate", "amplitude", "z", "date"]
    num_request_bands = len(request_band_names)

    Q, R, P = kalman_params["Q"], kalman_params["R"], kalman_params["P"]
    Q = np.diag([float(x) for x in Q])

    R = np.array([float(R)]).reshape(NUM_MEASURES, NUM_MEASURES)
    P_arr = np.diag([float(x) for x in P])

    P = ee.Image(ee.Array(P_arr.tolist())).rename("P")

    H = utils.sinusoidal(
        len([x for x in modality.keys() if modality[x] and x.endswith("modal")]),
        modality["linear"],
        modality["constant"],
    )

    kalman_init = {
        "F": utils.identity(num_params),
        "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
        "H": H,
        "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
        "num_params": num_params,
    }

    write_json(
        {
            "process_noise": Q.tolist(),
            "measurement_noise": R.tolist(),
            "initial_state_covariance": P_arr.tolist(),
        },
        os.path.join(SCRIPT_DIR, "eeek_params.json"),
    )

    points = parse_points(POINTS_FILE)

    ###########################################################
    ##### Run Kalman filter and bulcd across all points #######
    ###########################################################
    @retry.Retry()
    def process_point(kwargs):
        index = kwargs["index"]
        coords = (float(kwargs["longitude"]), float(kwargs["latitude"]))

        parameters = run_specific_parameters(study_area=ee.Geometry.Point(coords))

        organized_inputs = organize_inputs(parameters)

        result = kalman_with_bulcd(organized_inputs)

        pprint(result["multi_band_bulc_return"].getInfo())

        # x0 = ee.Image(
        #     ee.Array(np.array(kwargs["x0"]).reshape(num_params, NUM_MEASURES).tolist())
        # ).rename("x")

        # kalman_init["init_image"] = ee.Image.cat([P, x0])
        # kalman_init["point_coords"] = coords

        # kalman_result = kalman_filter.kalman_filter(
        #     organized_inputs["kalman_with_bulcd_params"], **kalman_init
        # )

        # states = (
        #     kalman_result.map(lambda im: utils.unpack_arrays(im, param_names))
        #     .select(request_band_names)
        #     .toBands()
        # )

        # request = utils.build_request(coords)
        # request["expression"] = states
        # data = utils.compute_pixels_wrapper(request).reshape(-1, num_request_bands)

        # df = pd.DataFrame(data, columns=request_band_names)
        # df["point"] = [index] * df.shape[0]

        # # put point as the first column
        # df = df[["point"] + request_band_names]

        # basename, ext = os.path.splitext(args["output"])
        # shard_path = basename + f"-{index:06d}" + ext
        # df.to_csv(shard_path, index=False)

        # return shard_path

    # with ProcessPool(nodes=40) as pool:
    #     all_output_files = pool.map(process_point, points)

    result = process_point(points[0])
    # all_output_files = [result]

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    # all_results = pd.concat(map(pd.read_csv, all_output_files), ignore_index=True)
    # all_results.to_csv(args["output"], index=False)

    # for f in all_output_files:
    #     os.remove(f)


if __name__ == "__main__":
    main()
