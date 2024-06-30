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

ee.Initialize()

from pprint import pprint

from eeek.gather_collections import afn_gather_collections_and_reduce


def scale_landsat8(image):
    """Based on https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2"""
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


L8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    # .filter(ee.Filter.lte("CLOUD_COVER", 30))
    .map(scale_landsat8)
    .select("SR_B6")
)

EXPORTED_LANDSAT8_FROM_JS = (
    ee.ImageCollection("projects/api-project-269347469410/assets/kalman-value-collection-60-180")
)


L8_GATHER_COLLECTIONS = afn_gather_collections_and_reduce({
    "L8dictionary": {
        "years_list": [2020],
        "first_doy": 1,
        "last_doy": 365,
        "cloud_cover_threshold": 30
    },
    "default_study_area": (
        ee.Geometry.Polygon([(-63.9533, -10.6813),(-63.9533, -10.1315), (-64.9118, -10.1315),(-64.9118, -10.6813)])
    ),
    "band_name_reduction": "swir",
    "which_reduction": "SWIR",
    "day_step_size": 4,
    "verbose": False,
    "dataset_selection": {
        "L5": False,
        "L7": False,
        "L8": True,
        "L9": False,
        "MO": False,
        "S2": False,
        "S1": False,
        "DW": False
    },
    "first_expectation_year": 2020,
    "verbose": False
})

COLLECTIONS = {
    "L8": L8,
    "EXPORTED_LANDSAT8_FROM_JS": EXPORTED_LANDSAT8_FROM_JS,
    "L8_GC": L8_GATHER_COLLECTIONS
}

if __name__ == "__main__":
    pprint(L8_GATHER_COLLECTIONS.getInfo())
