# First load the API file
import ee
import numpy as np
from lib.utils.ee.dates import convert_date
from lib.utils.ee.ccdc_utils import (
    get_multi_coefs,
    build_ccd_image,
)
from lib.utils import utils
import geemap.foliumap as geemap
from pprint import pprint

# Initialize the Earth Engine module
ee.Initialize()

# Load the results
ccdc = ee.Image("projects/GLANCE/RESULTS/CHANGEDETECTION/SA/Rondonia_example")

input_date = 1528675200000.0
date_params = {"input_format": 2, "input_date": input_date, "output_format": 1}
formatted_date = convert_date(date_params)

# Spectral band names. This list contains all possible bands in this dataset
# BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP"]
BANDS = ["SWIR1"]
# Names of the temporal segments
SEGS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

# Obtain CCDC results in 'regular' ee.Image format
ccdc_image = build_ccd_image(ccdc, len(SEGS), BANDS)

point = (-64.98237593840216, -12.969651493296622)

ccdc_image = ccdc_image.select(["S1_SWIR1_coef_.*"])

sample = ccdc_image.sample(ee.Geometry.Point(point), 10)

pprint(sample.getInfo())

coords = (point[0], point[1])
request = utils.build_request(coords)
request["expression"] = ccdc_image
ccdc_list = utils.compute_pixels_wrapper(request)

# Map = geemap.Map(center=[-64.98237593840216, -12.969651493296622], zoom=4)
# Map.addLayer(ccdc_image, {}, "ccdc_image")
# Map


# np.set_printoptions(suppress=True)
# print(ccdc_list[0:10])


# # Define bands to select.
# SELECT_BANDS = ["SWIR1"]

# # Define coefficients to select. This list contains all possible segments
# SELECT_COEFS = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3", "RMSE"]

# # Obtain coefficients
# coefs = get_multi_coefs(
#     ccdc_image, formatted_date, SELECT_BANDS, SELECT_COEFS, True, SEGS, "after"
# )


# coords = (point[0], point[1])
# request = utils.build_request(coords)
# request["expression"] = coefs
# coef_list = utils.compute_pixels_wrapper(request)

# print(coef_list)
