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


def get_harmonic_trend_coefficients(options):
    harmonic_trend = options.get("harmonic_trend", None)

    assert (
        harmonic_trend is not None
    ), "harmonic trend coefficients are required for generating kalman estimate vs harmonic trend plot"

    coef_dic = dict()

    with open(harmonic_trend, "r") as file:
        reader = csv.reader(file)

        for line in reader:
            coef_dic[line[0]] = line[1:]

    return coef_dic


def get_harmonic_trend_estimates(harmonic_trend_coefs, frac_of_year, harmonic_flags):
    harmonic_params, _ = parse_harmonic_params(harmonic_flags)

    harmonic_trend_coefs_by_year = []

    for frac_year in list(frac_of_year):
        year = int(frac_year)
        coefs = harmonic_trend_coefs.get(str(year), [])

        invalid_coefs = False

        if len(coefs) != len(harmonic_params):
            invalid_coefs = True

        index = 0

        args = {}

        if harmonic_flags.get(Harmonic.INTERCEPT.value, False):
            args[Harmonic.INTERCEPT.value] = 0 if invalid_coefs else coefs[index]
            index += 1

        if harmonic_flags.get(Harmonic.SLOPE.value, False):
            args[Harmonic.SLOPE.value] = 0 if invalid_coefs else coefs[index]
            index += 1

        args[Harmonic.COS.value] = 0 if invalid_coefs else coefs[index]
        index += 1

        args[Harmonic.SIN.value] = 0 if invalid_coefs else coefs[index]
        index += 1

        if harmonic_flags.get(Harmonic.BIMODAL.value, False):
            args[Harmonic.COS2.value] = 0 if invalid_coefs else coefs[index]
            index += 1

            args[Harmonic.SIN2.value] = 0 if invalid_coefs else coefs[index]
            index += 1

        if harmonic_flags.get(Harmonic.TRIMODAL.value, False):
            args[Harmonic.COS3.value] = 0 if invalid_coefs else coefs[index]
            index += 1

            args[Harmonic.SIN3.value] = 0 if invalid_coefs else coefs[index]
            index += 1

        harmonic_trend_coefs_by_year.append((frac_year, args))

    estimates = [
        [
            frac_year,
            calculate_harmonic_estimate(coefs, frac_year),
        ]
        for frac_year, coefs in harmonic_trend_coefs_by_year
    ]

    return pd.DataFrame(estimates, columns=[FRACTION_OF_YEAR, Harmonic.FIT.value])


def kalman_estimate_vs_harmonic_trend_plot(
    axs,
    data,
    options,
):
    harmonic_trend_coefs = get_harmonic_trend_coefficients(options)
    harmonic_fit_df = get_harmonic_trend_estimates(
        harmonic_trend_coefs,
        data[FRACTION_OF_YEAR],
        options.get("harmonic_flags", {}),
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
