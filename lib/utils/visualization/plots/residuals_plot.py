import json
import pandas as pd
import matplotlib.dates as mdates

from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import DATE_LABEL


def residuals_plot(
    axs,
    data,
    options,
):
    residuals_path = options.get("residuals_path", None)

    if residuals_path is None:
        print("No residuals path provided")
        return

    residuals = pd.read_csv(residuals_path)

    residuals[DATE_LABEL] = pd.to_datetime(residuals[DATE_LABEL])

    axs.scatter(
        residuals[DATE_LABEL],
        residuals["kalman_residual"],
        label="Kalman Residual",
        color="blue",
        marker="o",
        s=10,
    )

    axs.scatter(
        residuals[DATE_LABEL],
        residuals["ccdc_residual"],
        label="CCDC Residual",
        color="green",
        marker="o",
        s=10,
    )

    axs.axhline(0, color="black", linestyle="--")

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", "Residuals"))
