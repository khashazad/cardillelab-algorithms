"""
Defines standard functions for x, P, F, Q, H, and R.
"""
import math

import ee

from eeek import constants

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


def from_band_transposed(band_name, n):
    """Creates an array image with curr[band_name] stacked n times

    Useful to get R from a band e.g. cloud score plus.

    Args:
        band_name: str, band in curr to populate array with
        n: int, shape of resulting array will be (n, 1)

    Returns:
        function: ee.Image, dict -> ee.Image
    """

    def inner(curr, **kwargs):
        return curr.select(band_name).toArray().arrayRepeat(1, n).matrixTranspose()

    return inner


def from_band_diagonal(curr, band_name, n):
    """Creates an array image with curr[band_name] repeated along the diagonal.

    Args:
        band_name: str, band in curr to populate array with
        n: int, shape of resulting array will be (n, n)

    Returns:
        function: ee.Image, dict -> ee.Image
    """

    def inner(curr, **kwargs):
        return (
            curr.select(band_name)
            .toArray()
            .arrayRepeat(1, n)
            .matrixTranspose()
            .matrixToDiag()
        )

    return inner


def sinusoidal(num_params, linear_term=False):
    """Creates sinusoid function of the form a + b*cos(2pi*t) + c*sin(2pi*t)...

    Function will have an intercept but no linear term. cos is always paired
    with sin.

    Useful for H.

    Args:
        num_params: int, number of coefficients in the sinusoid function.

    Returns:
        function: ee.Image, dict -> ee.Image
    """

    def inner(t, **kwargs):
        bands = [ee.Image.constant(1.0)]
        if linear_term:  # TODO: if linear term proves useful drop if statement
            bands.append(t)
        for i in range((num_params - 1) // 2):
            freq = (i + 1) * 2 * math.pi
            bands.extend([t.multiply(freq).cos(), t.multiply(freq).sin()])
        image = ee.Image.cat(*bands).toArray(0)
        return image.arrayReshape(ee.Image(ee.Array([1, -1])), 2)

    return inner


def ccdc(t, **kwargs):
    """Creates a standard 8 parameter CCDC model.

    Useful for H.

    This is a more efficient implementation of sinusoidal(7, True)

    See: developers.google.com/earth-engine/datasets/catalog/GOOGLE_GLOBAL_CCDC_V1
    for model description.

    Args:
        None

    Returns:
        function () -> ee.Image
    """

    parameters = [
        ee.Image.constant(1.0),
        t,
        t.multiply(2 * math.pi).cos(),
        t.multiply(2 * math.pi).sin(),
        t.multiply(4 * math.pi).cos(),
        t.multiply(4 * math.pi).sin(),
        t.multiply(6 * math.pi).cos(),
        t.multiply(6 * math.pi).sin(),
    ]
    image = ee.Image.cat(*parameters).toArray(0)
    return image.arrayReshape(ee.Image(ee.Array([1, -1])), 2).toFloat()


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


def unpack_arrays(image, param_names):
    """Unpack array image into separate bands.

    Can be mapped across the output of kalman_filter.

    Currently only unpacks x and P. Names covariance bands as
    cov_{param1}_{param2}

    Args:
        image: ee.Image
        param_names: list[str], the names to give each parameter in the state

    Returns:
        ee.Image with bands for each state variable, and the covariance between
        each pair of state variables.
    """
    x = image.select(constants.STATE).arrayProject([0]).arrayFlatten([param_names])
    P = image.select(constants.COV).arrayFlatten(
        [["cov_" + x for x in param_names], param_names]
    )
    return ee.Image.cat(x, P)


def prep_landsat_collection(region, start_date, end_date, max_cloud_cover, sensors=8):
    """Creates Landsat collection, applies scale factors, adds cloud score band.

    The cloud score is calculated using ee.Algorithms.Landsat.simpleCloudScore
    and is stored in a band named "cloud".

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
    filtered_toa_col = _filter(toa_col).map(ee.Algorithms.Landsat.simpleCloudScore)

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
