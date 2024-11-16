import pandas as pd
import matplotlib.dates as mdates

def bulc_probs(axes, actual, point_index, title):
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    prob_decrease = filtered_data["prob_decrease"]
    prob_stable = filtered_data["prob_no_change"]
    prob_increase = filtered_data["prob_increase"]
    dates = filtered_data["timestamp"]

    axes.plot(
        dates,
        prob_decrease,
        marker="o",
        label="Decrease",
        linestyle="-",
        color="red",
    )
    axes.plot(
        dates,
        prob_stable,
        marker="o",
        label="No Change",
        linestyle="-",
        color="green",
    )
    axes.plot(
        dates,
        prob_increase,
        marker="o",
        label="Increase",
        linestyle="-",
        color="blue",
    )

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.tick_params(axis="x", labelsize=8)

    axes.set_ylim(0, 1)

    axes.set_title(title) 