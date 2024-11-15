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
from typing import List

import ee
import numpy as np
import pandas as pd
import pyperclip

from kalman import kalman_filter
from kalman.kalman_helper import (
    parse_harmonic_params,
    parse_band_names,
    setup_kalman_init,
    unpack_kalman_results,
)
from lib import constants
from lib.utils import utils
from lib.constants import Harmonic, Kalman, KalmanRecordingFlags, Index
from lib.image_collections import COLLECTIONS
from lib.utils.ee.ccdc_utils import build_ccd_image, get_multi_coefs, build_segment_tag
from lib.utils.ee.dates import convert_date
import concurrent.futures

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


def append_ccdc_coefficients(kalman_output_path, points, index: Index, study_area):
    ccdc_asset = (
        COLLECTIONS["CCDC_Global"]
        .mosaic()
        .clip(ee.Geometry.Polygon(study_area["coords"]))
    )
    if index == Index.SWIR:
        bands = ["SWIR1"]
    else:
        bands = ["NIR", "SWIR1"]

    coefs = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
    segments_count = 10
    segments = build_segment_tag(segments_count)

    kalman_output_df = pd.read_csv(kalman_output_path)
    dates = kalman_output_df[kalman_output_df["point"] == 0]["timestamp"].to_numpy()

    ccdc_image = build_ccd_image(ccdc_asset, segments_count, bands)

    ccdc_coefs_ic = ee.ImageCollection([])

    for date in dates:
        formatted_date = convert_date(
            {"input_format": 2, "input_date": date, "output_format": 1}
        )

        ccdc_coefs_image = get_multi_coefs(
            ccdc_image, formatted_date, bands, coefs, segments, segments, "after"
        )

        ccdc_coefs_ic = ccdc_coefs_ic.merge(ee.ImageCollection([ccdc_coefs_image]))

    def process_point(i, point):
        coords = (point[0], point[1])
        request = utils.build_request(coords)
        request["expression"] = ccdc_coefs_ic.toBands()
        coef_list = utils.compute_pixels_wrapper(request)

        ccdc_coefs_df = pd.DataFrame(
            coef_list.reshape(len(dates), len(coefs) * len(bands)),
            columns=[f"CCDC_{c}_{b}" for c in coefs for b in bands],
        )
        ccdc_coefs_df["timestamp"] = dates

        kalman_df = pd.DataFrame(kalman_output_df[kalman_output_df["point"] == i])

        kalman_ccdc_df = pd.merge(
            kalman_df,
            ccdc_coefs_df,
            on="timestamp",
            how="inner",
        )

        # Save to a temporary CSV file
        temp_file_path = (
            f"{os.path.dirname(kalman_output_path)}/outputs/{i:03d}_with_ccdc.csv"
        )
        kalman_ccdc_df.to_csv(temp_file_path, index=False)

        return temp_file_path

    with concurrent.futures.ThreadPoolExecutor() as executor:
        temp_files = list(executor.map(lambda p: process_point(*p), enumerate(points)))

    output_df = pd.concat(
        [pd.read_csv(temp_file) for temp_file in temp_files if temp_file is not None],
        ignore_index=True,
    )

    output_file_path = kalman_output_path.replace(".csv", "_with_ccdc.csv")
    output_df.to_csv(output_file_path, index=False)

    # clean up temporary files
    # for temp_file in temp_files:
    #     if temp_file is not None:
    #         os.remove(temp_file)


def main(
    kalman_parameters,
    value_collection,
    harmonic_flags: dict[Harmonic, any],
    recording_flags: dict[KalmanRecordingFlags, bool],
):
    band_names = parse_band_names(recording_flags, harmonic_flags)
    harmonic_params, _ = parse_harmonic_params(harmonic_flags)

    kalman_init = setup_kalman_init(kalman_parameters, harmonic_flags)

    kalman_result = kalman_filter.kalman_filter(
        collection=value_collection,
        init_image=kalman_init.get(Kalman.INITIAL_STATE.value),
        F=kalman_init.get(Kalman.F.value),
        Q=kalman_init.get(Kalman.Q.value),
        H=kalman_init.get(Kalman.H.value),
        R=kalman_init.get(Kalman.R.value),
        num_params=len(harmonic_params),
    )

    states = (
        kalman_result.map(
            lambda im: unpack_kalman_results(im, harmonic_params, recording_flags)
        )
        .select(band_names)
        .toBands()
    )

    return states
