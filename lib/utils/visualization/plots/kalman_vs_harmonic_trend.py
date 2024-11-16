import pandas as pd
import math
import numpy as np
import matplotlib.dates as mdates

from lib.utils.harmonic import calculate_harmonic_estimate
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT, FRAC_OF_YEAR
from lib.constants import (
    DATE,
    ESTIMATE,
    FRACTION_OF_YEAR,
    MEASUREMENT,
    TIMESTAMP,
    Kalman,
)


def get_harmonic_trend_coefficients(options):
    harmonic_trend = options.get("harmonic_trend", None)

    assert (
        harmonic_trend is not None
    ), "harmonic trend coefficients are required for generating kalman estimate vs harmonic trend plot"

    return pd.read_csv(harmonic_trend)


def get_harmonic_trend_estimates(harmonic_trend, timestamp):
    harmonic_trend[TIMESTAMP] = pd.to_datetime(timestamp, unit="ms")

    return harmonic_trend.groupby(harmonic_trend[TIMESTAMP].dt.year).size()


def kalman_estimate_vs_harmonic_trend(
    axs,
    data,
    options,
):
    harmonic_trend = get_harmonic_trend_coefficients(options)

    print(data[DATE])
    input()
    data["harmonic_trend"] = calculate_harmonic_estimate(
        harmonic_trend, data[FRACTION_OF_YEAR]
    )

    data[DATE] = pd.to_datetime(data[DATE])

    axs.plot(
        data[DATE],
        data[ESTIMATE],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE],
        data["harmonic_trend"],
        label="Harmonic Trend",
        linestyle="--",
        color="green",
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
