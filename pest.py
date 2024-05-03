"""Runs the Kalman Filter for PEST parameter estimation.

The input file, containing the parameters that PEST is estimating, should
contain a list of comma separated numbers separated on each line. The first
line should contain the values for the Q matrix (the process noise), the second
line should contain the values for the R matrix (the measurement noise), the
third line should contain the values for the P matrix (the initial state
covariance). If 'seed_ccdc' is NOT set, a fourth line should be given and
contain the values of x (the initial state matrix).

The output file will contain a single number: the sum of the errors at all
points specified in the points file. The error is derived from comparing the
final kalman filter state to the final parameters of a CCDC run.

The points file should specify each location where you want to apply the Kalman
filter. Each line of the points file should contain 2 numbers: the longitude of
the point and the latitude of the point followed by two strings: the date to
start running the Kalman filter and the date to stop running the kalman filter
(in YYYY-MM-dd format).

The target_ccdc file should contain the parameters for ccdc_utils.get_ccdc_coefs
as key,value pairs, see ccdc_utils.parse_ccdc_params for a more complete
explanation.

If seed_ccdc is give, it should have the same format as target_ccdc, but for
the CCDC image to seed the Kalman filter with.

All points specified in the points file must overlap with the target CCDC image
and the seed CCDC image (if one is given).
"""
import argparse

import ee
import numpy as np

from eeek import kalman_filter, utils, ccdc_utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

NUM_PARAMS = 8  # CCDC has 8 parameters
NUM_MEASURES = 1  # eeek only supports one band at a time


def main(args):
    #################################################
    ########### Read in PEST parameters #############
    #################################################
    with open(args.input, "r") as f:
        lines = f.readlines()

        Q, R, P = lines[:3]
        Q = np.array([float(x) for x in Q.split(",")]).reshape(NUM_PARAMS, NUM_PARAMS)
        R = np.array([float(x) for x in R.split(",")]).reshape(
            NUM_MEASURES, NUM_MEASURES
        )
        P = np.array([float(x) for x in P.split(",")]).reshape(NUM_PARAMS, NUM_PARAMS)
        P = ee.Image(ee.Array(P.tolist())).rename("P")

        if args.seed_ccdc is None:
            assert (
                len(lines) >= 4
            ), "must give x0 in PEST input file when initial_x=='pest'"
            x0 = lines[3]
            x0 = np.array([float(x) for x in x0.split(",")]).reshape(
                NUM_PARAMS, NUM_MEASURES
            )
            x0 = ee.Image(ee.Array(x0.tolist())).rename("x")
        else:
            ccdc_seed_args = ccdc_utils.parse_ccdc_params(args.seed_ccdc)
            seed = ccdc_utils.get_ccdc_coefs(**ccdc_seed_args)
            x0 = seed.toArray().toArray(1).rename("x")

        init = {
            "init_image": ee.Image.cat([P, x0]),
            "F": utils.identity(NUM_PARAMS),
            "Q": lambda **kwargs: ee.Image(ee.Array(Q.tolist())),
            "H": utils.ccdc,
            "R": lambda **kwargs: ee.Image(ee.Array(R.tolist())),
            "num_params": NUM_PARAMS,
        }

    #################################################
    ############## Run Kalman filter ################
    #################################################

    error = ee.Number(0)
    with open(args.points, "r") as f:
        for line in f.readlines():
            lon, lat, start, stop = line.split(",")
            point = ee.Geometry.Point((float(lon), float(lat)))
            col = utils.prep_landsat_collection(
                point, start.strip(), stop.strip(), args.max_cloud_cover
            )

            kalman_result = kalman_filter.kalman_filter(col, **init)
            kalman_result = kalman_result.map(
                lambda im: utils.unpack_arrays(im, ccdc_utils.HARMONIC_TAGS)
            )
            final_kalman_result = ee.Image(
                kalman_result.toList(1, kalman_result.size().subtract(1)).get(0)
            ).select(ccdc_utils.HARMONIC_TAGS)

            ccdc_target_args = ccdc_utils.parse_ccdc_params(args.target_ccdc)
            target = ccdc_utils.get_ccdc_coefs(**ccdc_target_args)

            comparison = final_kalman_result.spectralDistance(
                target,
                args.comparison_metric,
            ).rename("error")

            error = error.add(
                comparison.sample(region=point, numPixels=1).first().getNumber("error")
            )

    #################################################
    ############ Save results for PEST ##############
    #################################################

    with open(args.output, "w") as f:
        f.write(f"{error.getInfo()}\n")


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
        "--seed_ccdc",
        help="CCDC parameter file to use as initial state",
    )
    parser.add_argument(
        "--target_ccdc",
        required=True,
        help="CCDC parameter file to compare final state against",
    )
    parser.add_argument(
        "--max_cloud_cover",
        default=30,
        type=int,
        help="filter images with cloud cover > max_cloud_cover",
    )
    parser.add_argument(
        "--band_name",
        default="SWIR1",
        help="landsat band name to run kalman filter over",
    )
    parser.add_argument(
        "--comparison_metric",
        default="sam",
        choices=["sam", "sid", "sed", "emd"],
        help="spectral difference metric used to compare final state",
    )
    main(parser.parse_args())
