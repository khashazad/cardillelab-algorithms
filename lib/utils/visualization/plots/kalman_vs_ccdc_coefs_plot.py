from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE,
    Harmonic,
    CCDC,
)


def kalman_vs_ccdc_coefs_plot(
    axs,
    data,
    options,
):
    ccdc_coef = lambda coef: f"{CCDC.BAND_PREFIX.value}_{coef}"

    harmonic_flags = options.get("harmonic_flags", {})

    data[DATE] = pd.to_datetime(data[DATE])

    columns = data.columns

    if harmonic_flags.get(Harmonic.INTERCEPT.value, False):
        if Harmonic.INTERCEPT.value in columns:
            axs.plot(
                data[DATE],
                data[Harmonic.INTERCEPT.value],
                label="Intercept",
                linestyle="-",
                color="#000000",
            )

        if ccdc_coef(Harmonic.SLOPE.value) in columns:
            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.INTERCEPT.value)],
                label="CCDC Intercept",
                linestyle="--",
                color="#000000",
            )

    if harmonic_flags.get(Harmonic.SLOPE.value, False):
        if Harmonic.SLOPE.value in columns:
            axs.plot(
                data[DATE],
                data[Harmonic.SLOPE.value],
                label="Slope",
                linestyle="-",
                color="#808080",
            )

        if ccdc_coef(Harmonic.SLOPE.value) in columns:
            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.SLOPE.value)],
                label="CCDC Slope",
                linestyle="--",
                color="#808080",
            )

    if harmonic_flags.get(Harmonic.UNIMODAL.value, False):
        if Harmonic.COS.value in columns and Harmonic.SIN.value in columns:
            axs.plot(
                data[DATE],
                data[Harmonic.COS.value],
                label="Cos",
                linestyle="-",
                color="#1E90FF",
            )

            axs.plot(
                data[DATE],
                data[Harmonic.SIN.value],
                label="Sin",
                linestyle="-",
                color="#FF4500",
            )

        if (
            ccdc_coef(Harmonic.COS.value) in columns
            and ccdc_coef(Harmonic.SIN.value) in columns
        ):
            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.COS.value)],
                label="CCDC Cos",
                linestyle="--",
                color="#1E90FF",
            )

            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.SIN.value)],
                label="CCDC Sin",
                linestyle="--",
                color="#FF4500",
            )

    if harmonic_flags.get(Harmonic.BIMODAL.value, False):
        if Harmonic.COS2.value in columns and Harmonic.SIN2.value in columns:
            axs.plot(
                data[DATE],
                data[Harmonic.COS2.value],
                label="Cos2",
                linestyle="-",
                color="#32CD32",
            )

            axs.plot(
                data[DATE],
                data[Harmonic.SIN2.value],
                label="Sin2",
                linestyle="-",
                color="#FFA500",
            )

        if (
            ccdc_coef(Harmonic.COS2.value) in columns
            and ccdc_coef(Harmonic.SIN2.value) in columns
        ):
            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.COS2.value)],
                label="CCDC Cos2",
                linestyle="--",
                color="#32CD32",
            )

            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.SIN2.value)],
                label="CCDC Sin2",
                linestyle="--",
                color="#FFA500",
            )

    if harmonic_flags.get(Harmonic.TRIMODAL.value, False):
        if Harmonic.COS3.value in columns and Harmonic.SIN3.value in columns:
            axs.plot(
                data[DATE],
                data[Harmonic.COS3.value],
                label="Cos3",
                linestyle="-",
                color="#8A2BE2",
            )

            axs.plot(
                data[DATE],
                data[Harmonic.SIN3.value],
                label="Sin3",
                linestyle="-",
                color="#00CED1",
            )

        if (
            ccdc_coef(Harmonic.COS3.value) in columns
            and ccdc_coef(Harmonic.SIN3.value) in columns
        ):
            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.COS3.value)],
                label="CCDC Cos3",
                linestyle="--",
                color="#8A2BE2",
            )

            axs.plot(
                data[DATE],
                data[ccdc_coef(Harmonic.SIN3.value)],
                label="CCDC Sin3",
                linestyle="--",
                color="#00CED1",
            )

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", ""))
