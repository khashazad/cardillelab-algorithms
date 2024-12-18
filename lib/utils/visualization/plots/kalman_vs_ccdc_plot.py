import json
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE_LABEL,
    ESTIMATE_LABEL,
    Harmonic,
    Kalman,
    CCDC,
)
from lib.utils.visualization.constant import COLOR_PALETTE_10
from lib.utils.visualization.vis_utils import plot_ccdc_segments


def kalman_vs_ccdc_plot(
    axs,
    data,
    options,
):
    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    # ccdc_coef = lambda coef: f"{CCDC.BAND_PREFIX.value}_{coef}"

    # ccdc_filtered = data[
    #     (data[ccdc_coef(Harmonic.INTERCEPT.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.SLOPE.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.COS.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.SIN.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.COS2.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.SIN2.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.COS3.value)] != 0)
    #     | (data[ccdc_coef(Harmonic.SIN3.value)] != 0)
    # ]

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_LABEL],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE_LABEL],
        data[CCDC.FIT.value],
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

    if options.get(CCDC.SEGMENTS.value, False):
        plot_ccdc_segments(axs, data, options[CCDC.SEGMENTS.value])

    axs.axvline(
        x=pd.Timestamp(year=2020, month=1, day=1),
        color="red",
        linestyle="dashdot",
        label="ccdc",
    )

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", ""))
