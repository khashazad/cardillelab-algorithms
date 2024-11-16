import pandas as pd
import matplotlib.dates as mdates

def residuals_over_time(axes, actual, expected, point_index, title, fixed_y_axis=False):
    actual = actual.merge(expected, on=["point", "timestamp"], suffixes=("", "_target"))

    actual["timestamp"] = pd.to_datetime(actual["timestamp"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    intercept_residual = filtered_data["INTP"] - filtered_data["intercept"]
    cos_residual = filtered_data["COS0"] - filtered_data["cos"]
    sin_residual = filtered_data["SIN0"] - filtered_data["sin"]
    dates = filtered_data["timestamp"]

    axes.scatter(
        dates, intercept_residual, label="intercept", linestyle="-", color="blue", s=10
    )
    axes.scatter(dates, cos_residual, label="cos", linestyle="-", color="green", s=10)
    axes.scatter(dates, sin_residual, label="sin", linestyle="-", color="red", s=13)

    axes.axhline(y=0, color="black", linestyle="--")

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.tick_params(axis="x", labelsize=8)

    axes.set_title(title) 