"""
Test that the pest interface runs successfully.
"""
from argparse import Namespace

import numpy as np

from pest import main

rng = np.random.default_rng()

ccdc_params = "\n".join([
    "raw_ccdc_image,users/parevalo_bu/ccdc_long",
    "segs,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10",
    "bands,SWIR1",
    "date,2022",
    "coef_tags,None",
])

points = "\n".join([
    "-64.49037,-10.46841,2019-01-01,2023-01-01",
    "-64.56727,-10.59481,2019-01-01,2023-01-01",
    "-64.42583,-10.40645,2019-01-01,2023-01-01",
])

inputs = "\n".join([
    ",".join([str(x) for x in rng.normal(0, 1, 64)]),
    "0.1234",
    ",".join([str(x) for x in rng.normal(0, 1, 64)]),
    ",".join([str(x) for x in rng.normal(0, 1, 8)]),
])


def test_pest_interface(tmpdir):
    directory = tmpdir.mkdir("pest")

    ccdc_param_file = directory.join("ccdc_params")
    ccdc_param_file.write(ccdc_params)

    points_file = directory.join("points")
    points_file.write(points)

    inputs_file = directory.join("inputs")
    inputs_file.write(inputs)

    output_file = directory.join("output")

    args = Namespace(
        input=inputs_file,
        output=output_file,
        points=points_file,
        target_ccdc=ccdc_param_file,
        seed_ccdc=None,
        max_cloud_cover=30,
        band_name="SWIR1",
        comparison_metric="sam",
    )

    main(args)

    output = float(output_file.read())

    assert isinstance(output, float)


