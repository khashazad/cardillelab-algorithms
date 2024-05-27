"""Code to generate arbitrary image collections.

Use this to change the specific parameters of the image collection that is used
as input to PEST.

Collections defined in this file should NOT have filterBounds or filterDate
applied them. Those filters will be applied inside pest_eeek.py to allow each
point defined in the pest points file to have a different date range and be in
a different location. eeek will run the kalman filter over the first band of
each image in the input image collection so you should always select the band
you want to use when defining the collection.

Ensure that each image collection you define in this file is set as a value in
the COLLECTIONS dictionary. 
To set which image collection is used in pest set the value of the
`--collection` flag to the key of the collection in COLLECTIONS that you want
to use for the given run.
"""

import ee

from eeek import constants


def scale_landsat8(image):
    """Based on https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2"""
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


L8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filter(ee.Filter.lte("CLOUD_COVER", 30))
    .map(scale_landsat8)
    .select("SR_B7")
)

COLLECTIONS = {
    "L8": L8,
}
