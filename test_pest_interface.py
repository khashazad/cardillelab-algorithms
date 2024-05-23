"""
Test that the pest interface runs successfully.
"""
from argparse import Namespace

import numpy as np
import pandas as pd

from pest import main

rng = np.random.default_rng()

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

    points_file = directory.join("points")
    points_file.write(points)

    inputs_file = directory.join("inputs")
    inputs_file.write(inputs)

    output_file = directory.join("output")

    args = Namespace(
        input=inputs_file,
        output=output_file,
        points=points_file,
        include_intercept=True,
        include_slope=True,
        num_sinusoid_pairs=3,
        collection="L8",
    )

    main(args)

    result = pd.read_csv(output_file)

    target_columns = [
        "point", "INTP", "SLP", "COS0", "SIN0", "COS1", "SIN1", "COS2", "SIN2"
    ]

    # check that columns names are correct
    assert np.all(list(result) == target_columns)
