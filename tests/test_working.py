"""
Test various configurations of parameters are able to run successfully.
"""

import itertools
import math

import ee
import numpy as np
import pytest

from bulc import bulc
from kalman.kalman_filter import kalman_filter
from lib import constants
from utils.ee import ccdc_utils
from utils import utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

POINT = ee.Geometry.Point([-122.45, 37.78])
ROI = POINT.buffer(1024).bounds()
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
SCALE = 10
CLOUD_SCORE_PLUS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
TEST_PARAMS = list(itertools.product((1, 2, 3), (True, False), (True, False), (1,)))

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


def make_random_init(
    num_sinusoid_pairs, include_intercept, include_slope, num_measures
):
    num_params = (2 * num_sinusoid_pairs) + int(include_intercept) + int(include_slope)
    return {
        "init_image": ee.Image.cat(
            utils.constant_transposed(RNG.uniform(size=num_params).tolist())(),
            utils.constant_diagonal(RNG.uniform(size=num_params).tolist())(),
        ).rename([constants.STATE, constants.COV]),
        "F": utils.identity(num_params),
        "Q": utils.constant_diagonal(RNG.uniform(size=num_params).tolist()),
        "H": utils.sinusoidal(
            num_sinusoid_pairs,
            include_intercept=include_intercept,
            include_slope=include_slope,
        ),
        "R": utils.constant_transposed(RNG.uniform(size=num_measures).tolist()),
        "num_params": num_params,
    }


@pytest.mark.parametrize(
    "num_sinusoid_pairs,include_intercept,include_slope,num_measures", TEST_PARAMS[0:3]
)
def test_single_band_collection(
    num_sinusoid_pairs, include_intercept, include_slope, num_measures
):
    init = make_random_init(
        num_sinusoid_pairs, include_intercept, include_slope, num_measures
    )

    col = S2.filterBounds(POINT).select("B12").limit(20)

    verify_success(kalman_filter(col, **init))


@pytest.mark.parametrize(
    "num_sinusoid_pairs,include_intercept,include_slope,num_measures", TEST_PARAMS[3:6]
)
def test_multi_band_collection(
    num_sinusoid_pairs, include_intercept, include_slope, num_measures
):
    init = make_random_init(
        num_sinusoid_pairs, include_intercept, include_slope, num_measures
    )

    col = S2.filterBounds(POINT).limit(20)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


@pytest.mark.parametrize(
    "num_sinusoid_pairs,include_intercept,include_slope,num_measures", TEST_PARAMS[6:9]
)
def test_cloud_score_plus_as_measurement_noise(
    num_sinusoid_pairs, include_intercept, include_slope, num_measures
):
    init = make_random_init(
        num_sinusoid_pairs, include_intercept, include_slope, num_measures
    )
    init["R"] = utils.from_band_transposed("cloud", num_measures)

    col = utils.prep_sentinel_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


def test_scaled_band_value():
    init = make_random_init(1, True, True, 1)
    init["R"] = utils.from_band_transposed("cloud", 1, 0.1234)

    col = utils.prep_sentinel_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, measurement_band="B12", **init))


@pytest.mark.parametrize("sensors", [8, (9, 8), (7, 5)])
def test_simple_cloud_score_as_measurement_noise(sensors):
    init = make_random_init(2, False, True, 1)
    init["R"] = utils.from_band_transposed("cloud", 1)

    col = utils.prep_landsat_collection(POINT, "2020-01-01", "2021-01-01", 50)

    verify_success(kalman_filter(col, **init))


@pytest.mark.parametrize(
    "num_sinusoid_pairs,include_intercept,include_slope,num_measures", TEST_PARAMS[9:12]
)
def test_bulc_as_noise(
    num_sinusoid_pairs, include_intercept, include_slope, num_measures
):
    num_params = (2 * num_sinusoid_pairs) + int(include_intercept) + int(include_slope)
    init = make_random_init(
        num_sinusoid_pairs, include_intercept, include_slope, num_measures
    )
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
    num_measures = 1
    num_sinusoid_pairs = 1
    include_intercept = True
    include_slope = False
    num_params = (2 * num_sinusoid_pairs) + int(include_intercept) + int(include_slope)
    init = make_random_init(
        num_sinusoid_pairs, include_intercept, include_slope, num_measures
    )
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


def test_ccdc_as_H():
    init = make_random_init(3, True, True, 1)
    init["H"] = utils.ccdc

    col = S2.filterBounds(POINT).limit(20).select("B12")
    verify_success(kalman_filter(col, **init))


def test_ccdc_vs_sinusoidal():
    init1 = make_random_init(3, True, True, 1)
    init2 = init1.copy()

    init1["H"] = utils.ccdc
    init2["H"] = utils.sinusoidal(3, True, True)

    col = S2.filterBounds(POINT).limit(20).select("B12")
    verify_success(result1 := kalman_filter(col, **init1))
    verify_success(result2 := kalman_filter(col, **init2))

    result1 = result1.map(lambda im: utils.unpack_arrays(im,
        ccdc_utils.HARMONIC_TAGS))
    result2 = result2.map(lambda im: utils.unpack_arrays(im,
        ccdc_utils.HARMONIC_TAGS))

    bands = result1.first().bandNames()
    bands = bands.remove("x").remove("z").remove("P")

    result1 = result1.select(bands).toBands()
    result2 = result2.select(bands).toBands()

    result1 = result1.sample(
        region=POINT,
        scale=10,
        numPixels=1,
    ).first().getInfo()["properties"]
    result1 = np.array(list(result1.values()))

    result2 = result2.sample(
        region=POINT,
        scale=10,
        numPixels=1,
    ).first().getInfo()["properties"]
    result2 = np.array(list(result2.values()))

    assert np.allclose(result1, result2)


def test_sinusoidal():
    """Ensure sinusoidal creates the same coefs as hardcoded versions."""
    t = ee.Number(1.234)

    def _make_image(params):
        image = ee.Image.cat(*params).toArray(0)
        image = image.arrayReshape(ee.Image(ee.Array([1, -1])), 2).toFloat()
        return image

    def _make_comparison(im1, im2):
        region = ee.Geometry.Rectangle([[-10, 10], [10, -10]])
        val1 = (
            im1.sample(
                region=region,
                scale=10,
                numPixels=1,
            )
            .first()
            .get("array")
            .getInfo()
        )
        val2 = (
            im2.sample(
                region=region,
                scale=10,
                numPixels=1,
            )
            .first()
            .get("array")
            .getInfo()
        )
        return np.allclose(val1, val2)

    # compare 1 sinusoid pair, no slope, no intercept
    params = [
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
    ]
    image = _make_image(params)
    compare_image = utils.sinusoidal(1, False, False)(t)
    assert _make_comparison(image, compare_image)

    # compare 1 sinusoid pair, slope, intercept
    params = [
        ee.Image.constant(1.0),
        t,
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
    ]
    image = _make_image(params)
    compare_image = utils.sinusoidal(1, True, True)(t)
    assert _make_comparison(image, compare_image)

    # compare 3 sinusoid pairs, slope, no intercept
    params = [
        t,
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
        t.multiply(4 * math.pi).cos(),
        t.multiply(4 * math.pi).sin(),
        t.multiply(6 * math.pi).cos(),
        t.multiply(6 * math.pi).sin(),
    ]
    image = _make_image(params)
    compare_image = utils.sinusoidal(3, include_slope=True, include_intercept=False)(t)
    assert _make_comparison(image, compare_image)

    # compare 1 sinusoid pair, no slope, intercept
    params = [
        ee.Image.constant(1.0),
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
    ]
    image = _make_image(params)
    compare_image = utils.sinusoidal(1, include_slope=False, include_intercept=True)(t)
    assert _make_comparison(image, compare_image)


    # compare against ccdc
    image = utils.ccdc(t)
    compare_image = utils.sinusoidal(3, True, True)(t)
    assert _make_comparison(image, compare_image)
