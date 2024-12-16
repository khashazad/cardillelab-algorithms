"""
Defines standard functions for x, P, F, Q, H, and R.
"""

import io
import math

import numpy as np
import pandas as pd
from lib.constants import Kalman
from numpy.lib.recfunctions import structured_to_unstructured
import ee

from lib import constants

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

L9_SR = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
L8_SR = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7_SR = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
L5_SR = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
LANDSAT_SR = {9: L9_SR, 8: L8_SR, 7: L7_SR, 5: L5_SR}

L9_TOA = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")
L8_TOA = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
L7_TOA = ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA")
L5_TOA = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
LANDSAT_TOA = {9: L9_TOA, 8: L8_TOA, 7: L7_TOA, 5: L5_TOA}

S2_SR = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
S2_CLOUD_SCORE_PLUS = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")


def identity(num_params):
    """Creates an array image of the identity matrix with size num_params.

    Useful for F.

    Args:
        num_params: int

    Returns:
        function: dict -> ee.Image
    """

    def inner(**kwargs):
        return ee.Image(ee.Array.identity(num_params))

    return inner


def constant_diagonal(constant):
    """Creates an array image with constant along the diagonal.

    Useful for Q and P.

    Args:
        constant: list[number]

    Returns:
        function: dict -> ee.Image
    """

    def inner(**kwargs):
        return ee.Image(ee.Array([constant]).transpose().matrixToDiag())

    return inner


def constant_transposed(constant):
    """Creates an array image out of constant with shape (len(constant), 1)

    Useful for x and R.

    Args:
        constant: list[number]

    Returns:
        function: dict -> ee.Image
    """

    def inner(**kwargs):
        return ee.Image(ee.Array([constant]).transpose())

    return inner


def from_band_transposed(band_name, n, scale=None):
    """Creates an array image with curr[band_name] stacked n times

    Useful to get R from a band e.g. cloud score plus.

    Args:
        band_name: str, band in curr to populate array with
        n: int, shape of resulting array will be (n, 1)
        scale: list[number] used to scale the band values to allow different
            parameters/measurements to have different noise values, e.g., to
            prioritize updating one parameter over others.

    Returns:
        function: ee.Image, dict -> ee.Image
    """
    if scale is None:
        scale = [1.0] * n

    if not isinstance(scale, (list, tuple)):
        scale = [scale]

    assert len(scale) == n

    scale_im = constant_transposed(scale)()

    def inner(curr, **kwargs):
        output = curr.select(band_name).toArray().arrayRepeat(1, n).matrixTranspose()
        return output.multiply(scale_im)

    return inner


def from_band_diagonal(band_name, n, scale=None):
    """Creates an array image with curr[band_name] repeated along the diagonal.

    Args:
        band_name: str, band in curr to populate array with
        n: int, shape of resulting array will be (n, n)
        scale: list[number] used to scale the band values to allow different
            parameters/measurements to have different noise values, e.g., to
            prioritize updating one parameter over others.

    Returns:
        function: ee.Image, dict -> ee.Image
    """
    if scale is None:
        scale = [1.0] * n

    if not isinstance(scale, (list, tuple)):
        scale = [scale]

    assert len(scale) == n

    scale_im = constants_diagonal(scale)()

    def inner(curr, **kwargs):
        output = (
            curr.select(band_name)
            .toArray()
            .arrayRepeat(1, n)
            .matrixTranspose()
            .matrixToDiag()
        )
        return output.multiply(scale_im)

    return inner


def sinusoidal(num_sinusoid_pairs, include_slope=True, include_intercept=True):
    """Creates sinusoid function of the form a+b*t+c*cos(2pi*t)+d*sin(2pi*t)...

    Useful for H.

    Args:
        num_sinusoid_pairs: int, number of sine + cosine terms in the model.
        include_slope: bool, if True include a linear slope term in the model.
        include_intercept: bool, if True include a bias/intercept term in the
            model.

    Returns:
        function dict -> ee.Image
    """

    # becuase this function gets called once when building the Kalman Filter
    # but inner gets called at each update step of the Kalman Filter place all
    # of the expensive python if statements and for loops outside of inner so
    # that they are only executed once instead of at each time step
    multiply = []
    add = []
    t_selectors = []
    cosine_selectors = []
    sine_selectors = []
    num_params = 0
    if include_intercept:
        multiply.append(ee.Image.constant(0))
        add.append(ee.Image.constant(1))
        t_selectors.append(ee.Image.constant(1))
        cosine_selectors.append(ee.Image.constant(0))
        sine_selectors.append(ee.Image.constant(0))
        num_params += 1

    if include_slope:
        multiply.append(ee.Image.constant(1))
        add.append(ee.Image.constant(0))
        t_selectors.append(ee.Image.constant(1))
        cosine_selectors.append(ee.Image.constant(0))
        sine_selectors.append(ee.Image.constant(0))
        num_params += 1

    for i in range(num_sinusoid_pairs):
        freq = (i + 1) * 2 * math.pi
        multiply.extend([ee.Image.constant(freq)] * 2)
        add.extend([ee.Image.constant(0)] * 2)
        t_selectors.extend([ee.Image.constant(0)] * 2)
        cosine_selectors.extend([ee.Image.constant(1), ee.Image.constant(0)])
        sine_selectors.extend([ee.Image.constant(0), ee.Image.constant(1)])
        num_params += 2

    multiply = ee.Image.cat(*multiply)
    add = ee.Image.cat(*add)
    cosine_selectors = ee.Image.cat(*cosine_selectors)
    sine_selectors = ee.Image.cat(*sine_selectors)
    t_selectors = ee.Image.cat(*t_selectors)

    def inner(t, **kwargs):
        t = ee.Image.constant(ee.List.repeat(t, num_params))
        t = t.multiply(multiply).add(add)

        sine_terms = t.sin().multiply(sine_selectors)
        cosine_terms = t.cos().multiply(cosine_selectors)

        image = t.multiply(t_selectors).add(sine_terms).add(cosine_terms)

        image = image.toArray(0)
        return image.arrayReshape(ee.Image(ee.Array([1, -1])), 2)

    return inner


def ccdc(t, **kwargs):
    """Creates a standard 8 parameter CCDC model.

    Useful for H.

    This is a simpler implementation of sinusoidal(3, True, True) which is the
    default CCDC model.

    See: developers.google.com/earth-engine/datasets/catalog/GOOGLE_GLOBAL_CCDC_V1
    for model description.

    Args:
        t: ee.Number, number of years since START_DATE
        **kwargs: ignored, but passed to keep method signatures consistent

    Returns:
        ee.Image
    """

    parameters = [
        ee.Image.constant(1),
        t,
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
        t.multiply(4 * math.pi).cos(),
        t.multiply(4 * math.pi).sin(),
        t.multiply(6 * math.pi).cos(),
        t.multiply(6 * math.pi).sin(),
    ]
    image = ee.Image.cat(*parameters).toArray(0)
    return image.arrayReshape(ee.Image(ee.Array([1, -1])), 2)


def track_updated_measurement(x, H, **kwargs):
    """After updating the state recompute the measurement.

    Can be used as the postprocessing function.

    Args:
        x: ee.Image, the updated state
        H: function: dict -> ee.Image, the measurement function

    Returns:
        ee.Image
    """
    return H(**kwargs).matrixMultiply(x)


def prep_landsat_collection(
    region, start_date, end_date, max_cloud_cover=30, sensors=8
):
    """Creates Landsat collection, applies scale factors, adds cloud score band.

    The cloud score is calculated using ee.Algorithms.Landsat.simpleCloudScore
    and is stored in a band named "cloud". From the docs simpleCloudScore can
    only be calculated from the TOA collections so we fetch both the SR and TOA
    collections but return only the simpleCloudScore result from the TOA
    collection.

    Args:
        region: ee.Geometry, used in filterBounds
        start_date: str, used in filterDate
        end_date: str, used in filterDate
        max_cloud_cover: int, used in filter.lte("CLOUD_COVER", ...)
        sensors: int or list[int], which Landsat sensors to use

    Returns:
        ee.ImageCollection
    """

    if not isinstance(sensors, (list, tuple)):
        sensors = [sensors]

    for s in sensors:
        if s not in LANDSAT_SR.keys():
            raise NotImplementedError(
                f"Only Landsat {list(LANDSAT_SR.keys())} supported, got {sensors}"
            )

    def _filter(col):
        return (
            col.filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUD_COVER", max_cloud_cover))
        )

    def apply_scale_factors(image):
        optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(
            thermal_bands, None, True
        )

    sr_col = ee.ImageCollection(
        ee.FeatureCollection([LANDSAT_SR[k] for k in sensors]).flatten()
    )
    filtered_sr_col = _filter(sr_col).map(apply_scale_factors)

    toa_col = ee.ImageCollection(
        ee.FeatureCollection([LANDSAT_TOA[k] for k in sensors]).flatten()
    )
    filtered_toa_col = (
        _filter(toa_col)
        .map(ee.Algorithms.Landsat.simpleCloudScore)
        .map(lambda im: ee.Image(im).divide(100))
    )

    col = filtered_sr_col.linkCollection(filtered_toa_col, ["cloud"])

    return col.sort("system:time_start")


def prep_sentinel_collection(region, start_date, end_date, max_cloud_cover):
    """Creates Sentinel2 collection, adds cloud score band.

    The cloud score is based on the Sentinel Cloud Score + and is stored in a
    band named "cloud".

    Args:
        region: ee.Geometry, used in filterBounds
        start_date: str, used in filterDate
        end_date: str, used in filterDate
        max_cloud_cover: int, used in filter.lte("CLOUDY_PIXEL_PERCENTAGE", ...)

    Returns:
        ee.ImageCollection
    """

    col = (
        S2_SR.filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
    )

    cloud_score_col = (
        S2_CLOUD_SCORE_PLUS.filterBounds(region)
        .filterDate(start_date, end_date)
        .select("cs")
        .map(lambda im: im.rename("cloud"))
    )

    return col.linkCollection(cloud_score_col, ["cloud"])


def compute_pixels_wrapper(request):
    """Wraps ee.data.computePixels to allow loading larger npy files.

    Expects request to have fileFormat == "NPY"

    Args:
        request: dict, passed to ee.data.computePixels

    Returns:
        np.ndarray
    """
    result = ee.data.computePixels(request)

    return np.squeeze(
        structured_to_unstructured(np.load(io.BytesIO(result), allow_pickle=True))
    )


def get_pixels(coords, image):
    request = build_request(coords)
    request["expression"] = image
    return compute_pixels_wrapper(request)


def get_image_collection_pixels(coords, collection):

    band_count = ee.ImageCollection(collection).first().bandNames().size().getInfo()
    collection_size = ee.ImageCollection(collection).size().getInfo()
    chunks_size = 1024 // band_count

    chunks = []
    all_pixels = np.array([])

    collection_as_list = collection.toList(collection_size)

    for start in range(0, collection_size, chunks_size):
        end = min(start + chunks_size, collection_size)
        chunks.append(ee.ImageCollection(collection_as_list.slice(start, end)))

    for chunk in chunks:
        request = build_request(coords)
        request["expression"] = chunk.toBands()
        pixels = compute_pixels_wrapper(request)

        all_pixels = np.append(all_pixels, pixels)

    return all_pixels


def get_utm_from_lonlat(lon, lat):
    """Get the EPSG CRS Code for the UTM zone of a given lon, lat pait.

    Based on: https://stackoverflow.com/a/9188972

    Args:
        lon: float
        lat: float

    Returns:
        string
    """
    offset = 32601 if lat >= 0 else 32701
    return "EPSG:" + str(offset + (math.floor((lon + 180) / 6) % 60))


def build_request(point, scale=2):
    """Create a 1x1 numpy ndarray computePixels request at the given point.

    Args:
        point: (float, float), lat lon coordinates
        scale: int

    Returns:
        dict, passable to ee.data.computePixels, caller must set 'expression'
    """
    crs = get_utm_from_lonlat(*point)
    proj = ee.Projection(crs)
    geom = ee.Geometry.Point(point)
    coords = ee.Feature(geom).geometry(1, proj).getInfo()["coordinates"]
    request = {
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {
                "width": 1,
                "height": 1,
            },
            "affineTransform": {
                "scaleX": scale,
                "shearX": 0,
                "translateX": coords[0],
                "shearY": 0,
                "scaleY": -scale,
                "translateY": coords[1],
            },
            "crsCode": crs,
        },
    }
    return request
