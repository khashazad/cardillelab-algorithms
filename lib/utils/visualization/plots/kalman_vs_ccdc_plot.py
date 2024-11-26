from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE_LABEL,
    ESTIMATE_LABEL,
    FRACTION_OF_YEAR_LABEL,
    Harmonic,
    Kalman,
    CCDC,
)
from lib.utils.harmonic import calculate_harmonic_fit


def calculate_ccdc_fit(data):
    ccdc_coef = lambda coef: f"{CCDC.BAND_PREFIX.value}_{coef}"

    data["CCDC"] = data.apply(
        lambda row: calculate_harmonic_fit(
            {
                Harmonic.INTERCEPT.value: row[ccdc_coef(Harmonic.INTERCEPT.value)],
                Harmonic.SLOPE.value: row[ccdc_coef(Harmonic.SLOPE.value)],
                Harmonic.COS.value: row[ccdc_coef(Harmonic.COS.value)],
                Harmonic.SIN.value: row[ccdc_coef(Harmonic.SIN.value)],
                Harmonic.COS2.value: row[ccdc_coef(Harmonic.COS2.value)],
                Harmonic.SIN2.value: row[ccdc_coef(Harmonic.SIN2.value)],
                Harmonic.COS3.value: row[ccdc_coef(Harmonic.COS3.value)],
                Harmonic.SIN3.value: row[ccdc_coef(Harmonic.SIN3.value)],
            },
            row[FRACTION_OF_YEAR_LABEL],
        ),
        axis=1,
    )

    return data


def kalman_vs_ccdc_plot(
    axs,
    data,
    options,
):
    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    data = calculate_ccdc_fit(data)

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_LABEL],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE_LABEL],
        data["CCDC"],
        label="CCDC",
        linestyle="--",
        color="green",
    )

    axs.scatter(
        data[(data[Kalman.Z.value] != 0)][DATE_LABEL],
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
