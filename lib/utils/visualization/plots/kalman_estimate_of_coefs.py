import pandas as pd
import matplotlib.dates as mdates
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT


def kalman_estimate_of_coefficients(axs, data, expected, title, fixed_y_axis=False):
    data = data.merge(
        expected,
        on=["point", "timestamp"],
        suffixes=("", "_target"),
    )

    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

    dates = data["timestamp"]
    intp = data["INTP"]
    cos = data["COS0"]
    sin = data["SIN0"]

    target_intercept = data["intercept"]
    target_cos = data["cos"]
    target_sin = data["sin"]

    axs.plot(dates, intp, label="intercept", linestyle="-", color="blue")
    axs.plot(
        dates, target_intercept, label="target intercept", linestyle="--", color="blue"
    )

    axs.plot(dates, cos, label="cos", linestyle="-", color="green")
    axs.plot(dates, target_cos, label="target cos", linestyle="--", color="green")

    axs.plot(dates, sin, label="sin", linestyle="-", color="red")
    axs.plot(dates, target_sin, label="target sin", linestyle="--", color="red")

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if fixed_y_axis:
        axs.set_ylim(0, FIXED_Y_AXIS_LIMIT)

    axs.set_title(title)
