"""
Test various configurations of parameters are able to run successfully.
"""
import itertools

import numpy as np
import ee
import pytest
from eeek.kalman_filter import kalman_filter
from eeek import utils

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

POINT = ee.Geometry.Point([-122.45, 37.78])
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
CLOUD_SCORE_PLUS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
TEST_PARAMS = list(itertools.product((3, 5, 7), (1,)))

RNG = np.random.default_rng()


def make_random_init(num_params, num_measures):
    return {
        "init_x": utils.constant_transposed(RNG.uniform(size=num_measures).tolist())(),
        "init_P": utils.constant_diagonal(RNG.uniform(size=num_measures).tolist())(),
        "F": utils.identity(num_measures),
        "Q": utils.constant_diagonal(RNG.uniform(size=num_measures).tolist()),
        "H": utils.sinusoidal(num_params),
        "R": utils.constant_transposed(RNG.uniform(size=num_measures).tolist()),
        "num_params": num_params,
    }


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_single_band_collection(num_params, num_measures):
    init = make_random_init(num_params, num_measures)

    col = S2.filterBounds(POINT).select("B12").limit(20)

    result = kalman_filter(col, **init)

    assert result.size().getInfo() == 20


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_multi_band_collection(num_params, num_measures):
    init = make_random_init(num_params, num_measures)

    col = S2.filterBounds(POINT).limit(20)

    result = kalman_filter(col, measurement_band="B12", **init)

    assert result.size().getInfo() == 20


@pytest.mark.parametrize("num_params,num_measures", TEST_PARAMS)
def test_cloud_score_plus_as_measurement_noise(num_params, num_measures):
    init = make_random_init(num_params, num_measures)
    init["R"] = utils.from_band_transposed("cs", 3)

    col = S2.filterBounds(POINT).limit(20)
    col = col.linkCollection(CLOUD_SCORE_PLUS, ["cs"])
    result = kalman_filter(col, measurement_band="B12", **init)

    assert result.size().getInfo() == 20
