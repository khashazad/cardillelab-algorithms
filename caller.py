import os

import ee
import numpy as np
import pandas as pd
from google.api_core import retry
from pathos.pools import ProcessPool
from pprint import pprint

from kalman_with_bulcd.kalman_with_bulcd import kalman_with_bulcd
from utils.ee.image_compression_expansion import (
    convert_multi_band_image_to_image_collection,
)
from utils.filesystem import write_json
from utils import utils
from kalman_with_bulcd.parameters import run_specific_parameters
from kalman_with_bulcd.organize_inputs import organize_inputs
from IPython.display import display
from matplotlib import pyplot as plt

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


def merge_kalman_with_bulcd_results(
    bulc_probs, kalman_states, kalman_estimates, measurements, dates
):
    states_as_ic = convert_multi_band_image_to_image_collection(
        kalman_states, ["INTP", "COS0", "SIN0"]
    ).toList(300)
    estimates_as_ic = convert_multi_band_image_to_image_collection(
        kalman_estimates
    ).toList(300)
    measurements_as_ic = convert_multi_band_image_to_image_collection(
        measurements
    ).toList(300)
    dates_as_ic = convert_multi_band_image_to_image_collection(dates).toList(300)

    def merge_images(index, ic):
        img1 = ee.Image(estimates_as_ic.get(index))
        img2 = ee.Image(measurements_as_ic.get(index))
        img3 = ee.Image(dates_as_ic.get(index))

        return ee.ImageCollection(ic).merge(
            ee.ImageCollection.fromImages(
                [ee.Image.cat([img1, img2, img3]).rename(["estimate", "z", "date"])]
            )
        )

    return ee.ImageCollection(
        ee.List.sequence(0, estimates_as_ic.size().subtract(1)).iterate(
            merge_images,
            ee.ImageCollection.fromImages([]),
        )
    )


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
        "measurement_band": parameters["band_name_to_fit"],
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

        x0 = ee.Image(
            ee.Array(np.array(kwargs["x0"]).reshape(num_params, NUM_MEASURES).tolist())
        ).rename("x")

        kalman_init["init_image"] = ee.Image.cat([P, x0])

        organized_inputs["kalman_with_bulcd_params"]["kalman_params"] = kalman_init

        output = kalman_with_bulcd(organized_inputs)

        multi_band_bulc_return = output["multi_band_bulc_return"]
        all_bulc_layers = output["all_bulc_layers"]
        kalman_states = output["kalman_states"]
        kalman_covariances = output["kalman_covariances"]
        kalman_estimates = output["kalman_estimates"]
        kalman_measurements = output["kalman_measurements"]
        kalman_dates = output["kalman_dates"]
        final_bulc_probs = output["final_bulc_probs"]

        result = merge_kalman_with_bulcd_results(
            bulc_probs=final_bulc_probs,
            kalman_states=kalman_states,
            kalman_estimates=kalman_estimates,
            measurements=kalman_measurements,
            dates=kalman_dates,
        ).toBands()

        request = utils.build_request(coords)
        request["expression"] = result
        data = utils.compute_pixels_wrapper(request).reshape(-1, 3)

        df = pd.DataFrame(data, columns=["estimate", "z", "date"])
        df["point"] = [index] * df.shape[0]

        # put point as the first column
        df = df[["point"] + ["estimate", "z", "date"]]

        df = df[df["z"] != 0]

        basename, ext = os.path.splitext(os.path.join(RUN_DIR, "output.csv"))
        shard_path = basename + f"-{index:06d}" + ext
        df.to_csv(shard_path, index=False)

        df["date"] = pd.to_datetime(df["date"], unit="ms")

        plt.plot(df["date"], df["estimate"])
        plt.scatter(df["date"], df["z"])
        plt.show()

        return shard_path

    # with ProcessPool(nodes=40) as pool:
    # all_output_files = pool.map(process_point, points)

    result = process_point(points[0])
    all_output_files = [result]

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    # all_results = pd.concat(map(pd.read_csv, all_output_files), ignore_index=True)
    # all_results.to_csv(args["output"], index=False)

    # for f in all_output_files:
    #     os.remove(f)


if __name__ == "__main__":
    main()  #
