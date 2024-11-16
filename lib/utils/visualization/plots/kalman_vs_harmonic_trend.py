import pandas as pd
import math
import numpy as np
import matplotlib.dates as mdates
import csv

from lib.utils.harmonic import calculate_harmonic_estimate
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT, FRAC_OF_YEAR
from lib.constants import (
    DATE,
    ESTIMATE,
    FRACTION_OF_YEAR,
    MEASUREMENT,
    TIMESTAMP,
    Harmonic,
    Kalman,
)


def get_harmonic_trend_coefficients(options):
    harmonic_trend = options.get("harmonic_trend", None)

    assert (
        harmonic_trend is not None
    ), "harmonic trend coefficients are required for generating kalman estimate vs harmonic trend plot"

    coef_dic = dict()

    with open(harmonic_trend, "r") as file:
        reader = csv.reader(file)

        for line in reader:
            if len(line) == 4:
                coef_dic[line[0]] = [line[1], line[2], line[3]]

    return coef_dic


def get_harmonic_trend_estimates(harmonic_trend_coefs, frac_of_year):

    harmonic_trend_estimates = [
        [frac_of_year, *harmonic_trend_coefs.get(str(int(frac_of_year)))]
        for frac_of_year in list(frac_of_year)
    ]

    estimates = []

    for [frac_of_year, intercept, cos, sin] in harmonic_trend_estimates:
        estimates.append(
            [
                frac_of_year,
                calculate_harmonic_estimate(
                    {
                        Harmonic.INTERCEPT: intercept,
                        Harmonic.COS: cos,
                        Harmonic.SIN: sin,
                    },
                    frac_of_year,
                ),
            ]
        )

    return pd.DataFrame(estimates, columns=[FRACTION_OF_YEAR, Harmonic.FIT.value])


def kalman_estimate_vs_harmonic_trend(
    axs,
    data,
    options,
):
    harmonic_trend = get_harmonic_trend_coefficients(options)

    harmonic_fit_df = get_harmonic_trend_estimates(
        harmonic_trend, data[FRACTION_OF_YEAR]
    )

    data = data.merge(harmonic_fit_df, on=FRACTION_OF_YEAR, how="inner")

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
        data[Harmonic.FIT.value],
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
