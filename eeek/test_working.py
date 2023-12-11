"""
Test various configurations of parameters are able to run successfully.
"""
import itertools

import numpy as np
import ee
import pytest

from eeek.kalman_filter import kalman_filter
from eeek import utils, bulc, constants

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

POINT = ee.Geometry.Point([-122.45, 37.78])
ROI = POINT.buffer(1024).bounds()
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
SCALE = 10
CLOUD_SCORE_PLUS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
TEST_PARAMS = list(itertools.product((3, 5, 7), (1,)))

RNG = np.random.default_rng()


def verify_success(kalman_result, N=10):
    kalman_result_list = kalman_result.select(["x", "P"]).toList(kalman_result.size())
    final = ee.Image(kalman_result_list.get(-1))

    test = final.sample(
        region=ROI,
        scale=SCALE,
        numPixels=N,
    )

    assert test.size().getInfo() == N


def make_random_init(num_params, num_measures):
    return {
        "init_image": ee.Image.cat(
            utils.constant_transposed(RNG.uniform(size=num_params).tolist())(),
            utils.constant_diagonal(RNG.uniform(size=num_params).tolist())(),
        ).rename([constants.STATE, constants.COV]),
        "F": utils.identity(num_params),
        "Q": utils.constant_diagonal(RNG.uniform(size=num_params).tolist()),
        "H": utils.sinusoidal(num_params),
        "R": utils.constant_transposed(RNG.uniform(size=num_measures).tolist()),
        "num_params": num_params,
    }


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_single_band_collection(num_params, num_measures):
    init = make_random_init(num_params, num_measures)

    col = S2.filterBounds(POINT).select("B12").limit(20)

    verify_success(kalman_filter(col, **init))


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_multi_band_collection(num_params, num_measures):
    init = make_random_init(num_params, num_measures)

    col = S2.filterBounds(POINT).limit(20)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_cloud_score_plus_as_measurement_noise(num_params, num_measures):
    init = make_random_init(num_params, num_measures)
    init["R"] = utils.from_band_transposed("cloud", num_measures)

    col = utils.prep_sentinel_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


def test_scaled_band_value():
    num_params = 5
    num_measures = 1
    init = make_random_init(num_params, num_measures)
    init["R"] = utils.from_band_transposed("cloud", num_measures, 0.1234)

    col = utils.prep_sentinel_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


@pytest.mark.parametrize("sensors", [8, (9, 8), (7, 5)])
def test_simple_cloud_score_as_measurement_noise(sensors):
    num_params = 5
    num_measures = 1
    init = make_random_init(num_params, num_measures)
    init["R"] = utils.from_band_transposed("cloud", num_measures)

    col = utils.prep_landsat_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, **init))


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_bulc_as_noise(num_params, num_measures):
    init = make_random_init(num_params, num_measures)
    init["init_image"] = ee.Image.cat(
        utils.constant_transposed(RNG.uniform(size=num_params).tolist())(),
        utils.constant_diagonal(RNG.uniform(size=num_params).tolist())(),
        ee.Image.constant(ee.List.repeat(0, num_measures)).toFloat(),
        ee.Image(ee.Array(ee.List.repeat(1 / 3, 3))),
    ).rename(
        [
            constants.STATE,
            constants.COV,
            constants.RESIDUAL,
            constants.CHANGE_PROB,
        ]
    )
    init["preprocess_fn"] = bulc.preprocess(0.1)
    init["Q"] = bulc.bulc_as_noise(num_params)

    col = S2.filterBounds(POINT).limit(20).select("B12")
    verify_success(kalman_filter(col, **init))


def test_scaled_bulc():
    num_params = 5
    num_measures = 1
    init = make_random_init(num_params, num_measures)
    init["init_image"] = ee.Image.cat(
        utils.constant_transposed(RNG.uniform(size=num_params).tolist())(),
        utils.constant_diagonal(RNG.uniform(size=num_params).tolist())(),
        ee.Image.constant(ee.List.repeat(0, num_measures)).toFloat(),
        ee.Image(ee.Array(ee.List.repeat(1 / 3, 3))),
    ).rename(
        [
            constants.STATE,
            constants.COV,
            constants.RESIDUAL,
            constants.CHANGE_PROB,
        ]
    )
    init["preprocess_fn"] = bulc.preprocess(0.1)
    scale = RNG.uniform(size=num_params).tolist()
    init["Q"] = bulc.bulc_as_noise(num_params, scale)

    col = S2.filterBounds(POINT).limit(20).select("B12")
    verify_success(kalman_filter(col, **init))


@pytest.mark.parametrize("num_measures", [1])
def test_ccdc_as_H(num_measures):
    init = make_random_init(8, num_measures)
    init["H"] = utils.ccdc

    col = S2.filterBounds(POINT).limit(20).select("B12")
    verify_success(kalman_filter(col, **init))
