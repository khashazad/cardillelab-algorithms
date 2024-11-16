import pandas as pd
import numpy as np
import matplotlib.dates as mdates

def amplitude(axes, actual, expected, point_index, title, fixed_y_axis=False):
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], unit="ms")
    actual["expected_cos"] = expected["cos"]
    actual["expected_sin"] = expected["sin"]
    actual["expected_amplitude"] = np.sqrt(expected["cos"] ** 2 + expected["sin"] ** 2)

    filtered_data = actual[(actual["point"] == point_index)]

    cos = filtered_data["COS0"]
    sin = filtered_data["SIN0"]
    dates = filtered_data["timestamp"]

    amplitude_values = np.sqrt(cos**2 + sin**2)
    expected_amplitude = filtered_data["expected_amplitude"]

    axes.plot(
        dates,
        amplitude_values,
        label="Measured Amplitude",
        linestyle="-",
        color="purple",
    )
    axes.plot(
        dates,
        expected_amplitude,
        label="Expected Amplitude",
        linestyle="--",
        color="orange",
    )

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.tick_params(axis="x", labelsize=8)

    axes.set_title(title) 