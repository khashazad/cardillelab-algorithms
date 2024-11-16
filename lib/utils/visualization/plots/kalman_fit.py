from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np
import matplotlib.dates as mdates
import csv
from lib.utils.harmonic import parse_harmonic_params

from lib.utils.harmonic import calculate_harmonic_estimate
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT, FRAC_OF_YEAR
from lib.constants import (
    DATE,
    ESTIMATE,
    FRACTION_OF_YEAR,
    Harmonic,
    Kalman,
)


def kalman_fit(
    axs,
    data,
    options,
):
    data[DATE] = pd.to_datetime(data[DATE])

    axs.plot(
        data[DATE],
        data[ESTIMATE],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.scatter(
        data[(data[Kalman.Z.value] != 0)][DATE],
        data[(data[Kalman.Z.value] != 0)][Kalman.Z.value],
        label="Observed",
        s=13,
        color="red",
    )

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", ""))
