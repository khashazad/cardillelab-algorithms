import os

import ee
import numpy as np
import pandas as pd
from google.api_core import retry
from pathos.pools import ProcessPool
from pprint import pprint

from kalman_with_bulcd.kalman_with_bulcd import kalman_with_bulcd
from lib import constants
from lib.image_collections import COLLECTIONS
from utils.ee.gather_collections import reduce_collection_to_points_and_write_to_file
from utils.ee.image_compression_expansion import (
    convert_multi_band_image_to_image_collection,
)
from utils.filesystem import write_json, delete_and_create
from utils import utils
from kalman_with_bulcd.parameters import run_specific_parameters
from kalman_with_bulcd.organize_inputs import organize_inputs

from utils.prepare_optimization_run import (
    build_observations,
    create_points_file,
    fitted_coefficients_and_dates,
    parse_point_coordinates,
)
from utils.visualization.charts import generate_charts_single_run

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

POINT_SET = 10
POINTS_COUNT = 1
VERSION = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(
    SCRIPT_DIR,
    "runs",
    "kalman with bulcd",
    f"set {POINT_SET} - {POINTS_COUNT} points",
    f"v{VERSION}",
)
POINT_SET_DIRECTORY_PATH = f"{SCRIPT_DIR}/points/sets/{POINT_SET}"

delete_and_create(RUN_DIR)

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
    bulc_probs_drop_as_ic = convert_multi_band_image_to_image_collection(
        bulc_probs.select(".*probability_class1.*")
    ).toList(300)
    bulc_probs_no_change_as_ic = convert_multi_band_image_to_image_collection(
        bulc_probs.select(".*probability_class2.*")
    ).toList(300)
    bulc_probs_gain_as_ic = convert_multi_band_image_to_image_collection(
        bulc_probs.select(".*probability_class3.*")
    ).toList(300)

    states_as_ic = convert_multi_band_image_to_image_collection(
        kalman_states, ["INTP", "COS0", "SIN0"]
    ).toList(300).slice(1)
    estimates_as_ic = convert_multi_band_image_to_image_collection(
        kalman_estimates
    ).toList(300)
    measurements_as_ic = convert_multi_band_image_to_image_collection(
        measurements
    ).toList(300)
    dates_as_ic = convert_multi_band_image_to_image_collection(dates).toList(300)

    def merge_images(index, ic):
        cos = ee.Image(states_as_ic.get(index)).select("COS0")
        sin = ee.Image(states_as_ic.get(index)).select("SIN0")
        intp = ee.Image(states_as_ic.get(index)).select("INTP")
        estimate = ee.Image(estimates_as_ic.get(index))
        z = ee.Image(measurements_as_ic.get(index))
        date = ee.Image(dates_as_ic.get(index))
        prob_increase = ee.Image(bulc_probs_drop_as_ic.get(index))
        prob_no_change = ee.Image(bulc_probs_no_change_as_ic.get(index))
        prob_decrease = ee.Image(bulc_probs_gain_as_ic.get(index))

        return ee.ImageCollection(ic).merge(
            ee.ImageCollection.fromImages(
                [
                    ee.Image.cat(
                        [
                            intp,
                            cos,
                            sin,
                            estimate,
                            z,
                            date,
                            prob_increase,
                            prob_no_change,
                            prob_decrease,
                        ]
                    ).rename(
                        [
                            "INTP",
                            "COS0",
                            "SIN0",
                            "estimate",
                            "z",
                            "date",
                            "prob_increase",
                            "prob_no_change",
                            "prob_decrease",
                        ]
                    )
                ]
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

    request_band_names = [
        *param_names.copy(),
        "estimate",
        "z",
        "date",
        "prob_increase",
        "prob_no_change",
        "prob_decrease",
    ]
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
        os.path.join(RUN_DIR, "eeek_params.json"),
    )

    points = parse_points(os.path.join(RUN_DIR, "points.csv"))

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
        ).rename(constants.STATE)

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
        probability_layers = output["all_probability_layers"]

        result = merge_kalman_with_bulcd_results(
            bulc_probs=probability_layers,
            kalman_states=kalman_states,
            kalman_estimates=kalman_estimates,
            measurements=kalman_measurements,
            dates=kalman_dates,
        ).toBands()

        request = utils.build_request(coords)
        request["expression"] = result
        data = utils.compute_pixels_wrapper(request).reshape(-1, num_request_bands)

        df = pd.DataFrame(data, columns=request_band_names)
        df["point"] = [index] * df.shape[0]

        # put point as the first column
        df = df[["point"] + request_band_names]

        basename, ext = os.path.splitext(os.path.join(RUN_DIR, "output.csv"))
        shard_path = basename + f"-{index:06d}" + ext
        df.to_csv(shard_path, index=False)

        return shard_path

    # with ProcessPool(nodes=40) as pool:
    # all_output_files = pool.map(process_point, points)

    result = process_point(points[0])
    all_output_files = [result]

    #################################################
    ## Combine results from all runs to single csv ##
    #################################################
    all_results = pd.concat(map(pd.read_csv, all_output_files), ignore_index=True)
    all_results.to_csv(os.path.join(RUN_DIR, "output.csv"), index=False)

    for f in all_output_files:
        os.remove(f)


if __name__ == "__main__":
    if not os.path.exists(f"{RUN_DIR}/measurements.csv"):
        points = parse_point_coordinates(POINT_SET_DIRECTORY_PATH)
        reduce_collection_to_points_and_write_to_file(
            COLLECTIONS["L8_L9_2022_2023"],
            points,
            f"{RUN_DIR}/measurements.csv",
        )
        fitted_coefficiets_by_point = fitted_coefficients_and_dates(
            points, f"{RUN_DIR}/fitted_coefficients.csv"
        )

        create_points_file(f"{RUN_DIR}/points.csv", fitted_coefficiets_by_point)

        observations = build_observations(
            fitted_coefficiets_by_point, f"{RUN_DIR}/observations.csv"
        )

    main()

    generate_charts_single_run(
        f"{RUN_DIR}/output.csv",
        f"{RUN_DIR}/observations.csv",
        f"{RUN_DIR}/analysis",
        {
            "estimate": True,
            "final_2022_fit": False,
            "final_2023_fit": False,
            "intercept_cos_sin": True,
            "residuals": False,
            "amplitude": False,
            "bulc_probs": True,
        },
    )
