import pandas as pd
import matplotlib.dates as mdates
from utils.visualization.constant import FIXED_Y_AXIS_LIMIT

import numpy as np
import math


def calculate_ccdc_estimate(coeffs):
    def calculate_estimate(bands):
        phi = 2 * np.pi * coeffs["time"]
        phi_cos = phi.apply(math.cos)
        phi_sin = phi.apply(math.sin)

        output = (
            bands["CCDC_INTP"]
            + bands["CCDC_COS"] * phi_cos
            + bands["CCDC_SIN"] * phi_sin
        )

        if "CCDC_COS2" in bands.columns and "CCDC_SIN2" in bands.columns:
            phi_2 = 2 * phi * coeffs["time"]
            phi_cos_2 = phi_2.apply(math.cos)
            phi_sin_2 = phi_2.apply(math.sin)

            output = (
                output + bands["CCDC_COS2"] * phi_cos_2 + bands["CCDC_SIN2"] * phi_sin_2
            )

        if "CCDC_COS3" in bands.columns and "CCDC_SIN3" in bands.columns:
            phi_3 = 3 * phi * coeffs["time"]
            phi_cos_3 = phi_3.apply(math.cos)
            phi_sin_3 = phi_3.apply(math.sin)

            output = (
                output + bands["CCDC_COS3"] * phi_cos_3 + bands["CCDC_SIN3"] * phi_sin_3
            )

        if "CCDC_SLP" in bands.columns:
            output = output + bands["CCDC_SLP"] * coeffs["time"]

        return output

    swir_bands = coeffs.columns[coeffs.columns.str.endswith("SWIR1")]
    nir_bands = coeffs.columns[coeffs.columns.str.endswith("NIR")]

    swir = calculate_estimate(
        coeffs[swir_bands].rename(columns=lambda x: x.replace("_SWIR1", ""))
    )

    if len(nir_bands) > 0:
        nir = calculate_estimate(
            coeffs[nir_bands].rename(columns=lambda x: x.replace("_NIR", ""))
        )

        return nir - swir / nir + swir

    return swir


def estimate_vs_ccdc(axes, actual, point_index, title, fixed_y_axis=False):
    actual["time"] = (
        pd.to_datetime(actual["timestamp"], unit="ms") - pd.to_datetime("2016-01-01")
    ).dt.total_seconds() / (365.25 * 24 * 60 * 60)

    actual["ccdc_estimate"] = calculate_ccdc_estimate(actual)

    actual["timestamp"] = pd.to_datetime(actual["timestamp"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    axes.plot(
        filtered_data["timestamp"],
        filtered_data["estimate"],
        label="Estimate - Optimized",
        linestyle="-",
        color="blue",
    )
    axes.plot(
        filtered_data["timestamp"],
        filtered_data["ccdc_estimate"],
        label="CCDC",
        linestyle="--",
        color="green",
    )
    axes.scatter(
        filtered_data[(filtered_data["z"] != 0)]["timestamp"],
        filtered_data[(filtered_data["z"] != 0)]["z"],
        label="Observed",
        s=13,
        color="red",
    )

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.tick_params(axis="x", labelsize=8)

    if fixed_y_axis:
        axes.set_ylim(0, FIXED_Y_AXIS_LIMIT)

    axes.set_title(title)
