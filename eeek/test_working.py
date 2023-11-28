"""
Test various configurations of parameters are able to run successfully.
"""
import math

import ee
from eeek.kalman_filter import kalman_filter

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

POINT = ee.Geometry.Point([-122.45, 37.78])
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
CLOUD_SCORE_PLUS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
INIT_x = ee.Image(ee.Array([[0.04, 0.04, 0.03]]).transpose())
INIT_P = ee.Image(ee.Array([[0.04, 0.04, 0.03]]).transpose().matrixToDiag())


def F_fn(**kwargs):
    return ee.Image(ee.Array.identity(3))


def Q_fn(**kwargs):
    return ee.Image(ee.Array([[0.001, 0.0005, 0.0025]]).transpose().matrixToDiag())


def H_fn(t, **kwargs):
    t = t.multiply(2 * math.pi)
    H = ee.Image.cat(ee.Image.constant(1.0), t.cos(), t.sin()).toArray(0)
    return H.arrayReshape(ee.Image(ee.Array([1, -1])), 2)


def R_fn(**kwargs):
    return ee.Image(ee.Array([[0.1234]]))


def R_from_cloud_score_plus(curr, band="cs", **kwargs):
    shape = ee.Image(ee.Array([1, 1]))
    return curr.select(band).toArray().arrayReshape(shape, 2)


INIT = {
    "init_x": INIT_x,
    "init_P": INIT_P,
    "F": F_fn,
    "Q": Q_fn,
    "H": H_fn,
    "R": R_fn,
    "num_params": 3,
}


def test_single_band_collection():
    col = S2.filterBounds(POINT).select("B12").limit(20)
    result = kalman_filter(col, **INIT)
    assert result.size().getInfo() == 20


def test_multi_band_collection():
    col = S2.filterBounds(POINT).limit(20)
    result = kalman_filter(col, measurement_band="B12", **INIT)
    assert result.size().getInfo() == 20


def test_cloud_score_plus_as_measurement_noise():
    col = S2.filterBounds(POINT).limit(20)
    col = col.linkCollection(CLOUD_SCORE_PLUS, ["cs"])
    init = INIT.copy()
    init["R"] = R_from_cloud_score_plus
    result = kalman_filter(col, measurement_band="B12", **INIT)
    assert result.size().getInfo() == 20
