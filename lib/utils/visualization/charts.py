import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import enum

FIXED_Y_AXIS_LIMIT = 0.4
ASPECT_RATIO = (12, 8)


class ChartType(enum.Enum):
    KALMAN_VS_HARMONIC_FIT = "kalman vs harmonic fit"
    ESTIMATES_INTERCEPT_COS_SIN = "estimates intercept cos sin"
    RESIDUALS_OVER_TIME = "residuals over time"
    AMPLITUDE = "amplitude"
    BULC_PROBS = "bulc probs"
    KALMAN_VS_CCDC = "kalman vs ccdc"


CHART_TITLES = {
    ChartType.KALMAN_VS_HARMONIC_FIT: "Kalman vs Harmonic Fit",
    ChartType.ESTIMATES_INTERCEPT_COS_SIN: "Estimates Intercept Cos Sin",
    ChartType.RESIDUALS_OVER_TIME: "Residuals Over Time",
    ChartType.AMPLITUDE: "Amplitude",
    ChartType.BULC_PROBS: "Bulc Probs",
    ChartType.KALMAN_VS_CCDC: "Kalman vs CCDC",
}


def estimate(
    axes,
    actual,
    expected,
    point_index,
    title,
    fixed_y_axis=False,
    include_2022_fit=False,
    include_2023_fit=False,
):
    actual["time"] = (
        pd.to_datetime(actual["date"], unit="ms") - pd.to_datetime("2016-01-01")
    ).dt.total_seconds() / (365.25 * 24 * 60 * 60)

    phi = 6.283 * actual["time"]
    phi_cos = phi.apply(math.cos)
    phi_sin = phi.apply(math.sin)

    actual["observation_estimate"] = (
        expected["intercept"] + expected["cos"] * phi_cos + expected["sin"] * phi_sin
    )

    actual["date"] = pd.to_datetime(actual["date"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    if include_2022_fit:
        filtered_data_2022 = filtered_data[filtered_data["date"].dt.year == 2022]
        intercept = filtered_data_2022["INTP"].iloc[-1]
        cos0 = filtered_data_2022["COS0"].iloc[-1]
        sin0 = filtered_data_2022["SIN0"].iloc[-1]

        x = filtered_data["time"]

        A = np.sqrt(cos0**2 + sin0**2)
        phi = np.arctan2(sin0, cos0)

        y = intercept + A * np.cos(2 * np.pi * x - phi)

        axes.plot(filtered_data["date"], y, label="Final 2022 fit", color="orange")

    if include_2023_fit:
        filtered_data_2023 = filtered_data[filtered_data["date"].dt.year == 2023]
        intercept = filtered_data_2023["INTP"].iloc[-1]
        cos0 = filtered_data_2023["COS0"].iloc[-1]
        sin0 = filtered_data_2023["SIN0"].iloc[-1]

        x = filtered_data["time"]

        A = np.sqrt(cos0**2 + sin0**2)
        phi = np.arctan2(sin0, cos0)

        y = intercept + A * np.cos(2 * np.pi * x - phi)

        axes.plot(filtered_data["date"], y, label="Final 2023 fit", color="purple")

    axes.plot(
        filtered_data["date"],
        filtered_data["estimate"],
        label="Estimate - Optimized",
        linestyle="-",
        color="blue",
    )
    axes.plot(
        filtered_data["date"],
        filtered_data["observation_estimate"],
        label="Target",
        linestyle="--",
        color="green",
    )
    axes.scatter(
        filtered_data[(filtered_data["z"] != 0)]["date"],
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


def intercept_cos_sin(axes, actual, expected, point_index, title, fixed_y_axis=False):
    actual = actual.merge(
        expected,
        on=["point", "date"],
        suffixes=("", "_target"),
    )

    actual["date"] = pd.to_datetime(actual["date"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    dates = filtered_data["date"]
    intp = filtered_data["INTP"]
    cos = filtered_data["COS0"]
    sin = filtered_data["SIN0"]
    target_intercept = filtered_data["intercept"]
    target_cos = filtered_data["cos"]
    target_sin = filtered_data["sin"]

    axes.plot(dates, intp, label="intercept", linestyle="-", color="blue")
    axes.plot(
        dates, target_intercept, label="target intercept", linestyle="--", color="blue"
    )

    axes.plot(dates, cos, label="cos", linestyle="-", color="green")
    axes.plot(dates, target_cos, label="target cos", linestyle="--", color="green")

    axes.plot(dates, sin, label="sin", linestyle="-", color="red")
    axes.plot(dates, target_sin, label="target sin", linestyle="--", color="red")

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.tick_params(axis="x", labelsize=8)

    if fixed_y_axis:
        axes.set_ylim(0, FIXED_Y_AXIS_LIMIT)

    axes.set_title(title)


def residuals_over_time(axes, actual, expected, point_index, title, fixed_y_axis=False):
    actual = actual.merge(expected, on=["point", "date"], suffixes=("", "_target"))

    actual["date"] = pd.to_datetime(actual["date"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    intercept_residual = filtered_data["INTP"] - filtered_data["intercept"]
    cos_residual = filtered_data["COS0"] - filtered_data["cos"]
    sin_residual = filtered_data["SIN0"] - filtered_data["sin"]
    dates = filtered_data["date"]

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


def amplitude(axes, actual, expected, point_index, title, fixed_y_axis=False):
    actual["date"] = pd.to_datetime(actual["date"], unit="ms")
    actual["expected_cos"] = expected["cos"]
    actual["expected_sin"] = expected["sin"]
    actual["expected_amplitude"] = np.sqrt(expected["cos"] ** 2 + expected["sin"] ** 2)

    filtered_data = actual[(actual["point"] == point_index)]

    cos = filtered_data["COS0"]
    sin = filtered_data["SIN0"]
    dates = filtered_data["date"]

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


def bulc_probs(axes, actual, point_index, title):
    actual["date"] = pd.to_datetime(actual["date"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    prob_decrease = filtered_data["prob_decrease"]
    prob_stable = filtered_data["prob_no_change"]
    prob_increase = filtered_data["prob_increase"]
    dates = filtered_data["date"]

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


def calculate_ccdc_estimate(actual):
    phi = 2 * np.pi * actual["time"]
    phi_cos = phi.apply(math.cos)
    phi_sin = phi.apply(math.sin)

    output = (
        actual["CCDC_INTP"]
        + actual["CCDC_COS"] * phi_cos
        + actual["CCDC_SIN"] * phi_sin
    )

    if "CCDC_COS2" in actual.columns and "CCDC_SIN2" in actual.columns:
        phi_2 = 2 * phi * actual["time"]
        phi_cos_2 = phi_2.apply(math.cos)
        phi_sin_2 = phi_2.apply(math.sin)

        output = (
            output + actual["CCDC_COS2"] * phi_cos_2 + actual["CCDC_SIN2"] * phi_sin_2
        )

    if "CCDC_COS3" in actual.columns and "CCDC_SIN3" in actual.columns:
        phi_3 = 3 * phi * actual["time"]
        phi_cos_3 = phi_3.apply(math.cos)
        phi_sin_3 = phi_3.apply(math.sin)

        output = (
            output + actual["CCDC_COS3"] * phi_cos_3 + actual["CCDC_SIN3"] * phi_sin_3
        )

    if "CCDC_SLP" in actual.columns:
        output = output + actual["CCDC_SLP"] * actual["time"]

    return output


def estimate_vs_ccdc(axes, actual, point_index, title, fixed_y_axis=False):
    actual["time"] = (
        pd.to_datetime(actual["date"], unit="ms") - pd.to_datetime("2016-01-01")
    ).dt.total_seconds() / (365.25 * 24 * 60 * 60)

    actual["ccdc_estimate"] = calculate_ccdc_estimate(actual)

    actual["date"] = pd.to_datetime(actual["date"], unit="ms")

    filtered_data = actual[(actual["point"] == point_index)]

    axes.plot(
        filtered_data["date"],
        filtered_data["estimate"],
        label="Estimate - Optimized",
        linestyle="-",
        color="blue",
    )
    axes.plot(
        filtered_data["date"],
        filtered_data["ccdc_estimate"],
        label="CCDC",
        linestyle="--",
        color="green",
    )
    axes.scatter(
        filtered_data[(filtered_data["z"] != 0)]["date"],
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


def get_labels_and_handles(axs):
    handles, labels = axs.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    return unique_labels, unique_handles


def save_chart(fig, point_index, name, output_directory):
    point_directory = f"{output_directory}/points/{point_index}"
    os.makedirs(point_directory, exist_ok=True)
    fig.savefig(f"{point_directory}/{name}.png")

    directory = f"{output_directory}/{name}"
    os.makedirs(directory, exist_ok=True)
    fig.savefig(f"{directory}/{point_index}.png")


def generate_charts_comparing_runs(
    data_files, observation_file_path, output_directory, flags
):
    os.makedirs(output_directory, exist_ok=True)
    target_observations = pd.read_csv(observation_file_path)
    points_count = len(target_observations["point"].unique())

    for point_index in range(points_count):

        plots = []

        if flags["estimate"]:
            fig_estimate_vs_target, axs_estimate_vs_target = plt.subplots(
                2, 2, figsize=(12, 8)
            )
            plots.append(
                (fig_estimate_vs_target, axs_estimate_vs_target, "estimate vs target")
            )
        if flags["intercept_cos_sin"]:
            fig_intercept_cos_sin, axs_intercept_cos_sin = plt.subplots(
                2, 2, figsize=(12, 8)
            )
            plots.append(
                (fig_intercept_cos_sin, axs_intercept_cos_sin, "intercept cos sin")
            )
        if flags["residuals"]:
            fig_residuals, axs_residuals = plt.subplots(2, 2, figsize=(12, 8))
            plots.append((fig_residuals, axs_residuals, "residuals over time"))
        if flags["amplitude"]:
            fig_amplitude, axs_amplitude = plt.subplots(2, 2, figsize=(12, 8))
            plots.append((fig_amplitude, axs_amplitude, "amplitude"))

        for file_index, run_title in enumerate(data_files.keys()):
            eeek_output = pd.read_csv(data_files[run_title])

            for fig, axs, graph_type in plots:
                axes = axs[file_index // 2, file_index % 2]
                if graph_type == "estimate vs target":
                    estimate(
                        axes,
                        eeek_output.copy(),
                        target_observations.copy(),
                        point_index,
                        run_title,
                        include_2022_fit=flags["final_2022_fit"],
                        include_2023_fit=flags["final_2023_fit"],
                    )
                elif graph_type == "intercept cos sin":
                    intercept_cos_sin(
                        axes,
                        eeek_output.copy(),
                        target_observations.copy(),
                        point_index,
                        run_title,
                    )
                elif graph_type == "residuals over time":
                    residuals_over_time(
                        axes,
                        eeek_output.copy(),
                        target_observations.copy(),
                        point_index,
                        run_title,
                    )
                elif graph_type == "amplitude":
                    amplitude(
                        axes,
                        eeek_output.copy(),
                        target_observations.copy(),
                        point_index,
                        run_title,
                    )

        for fig, axs, graph_type in plots:
            labels, handles = get_labels_and_handles(axs[0, 0])
            fig.legend(handles, labels, loc="upper center", ncol=5)

            plt.tight_layout()
            save_chart(fig, point_index, graph_type, output_directory)
            plt.close(fig)


def generate_charts_single_run(
    data_file_path, observation_file_path, output_directory, flags
):
    os.makedirs(output_directory, exist_ok=True)
    target_observations = pd.read_csv(observation_file_path)
    points_count = len(target_observations["point"].unique())

    for point_index in range(points_count):

        plots = []

        if (
            ChartType.KALMAN_VS_HARMONIC_FIT in flags
            and flags[ChartType.KALMAN_VS_HARMONIC_FIT]
        ):
            fig_estimate_vs_target, axs_estimate_vs_target = plt.subplots(
                1, 1, figsize=ASPECT_RATIO
            )
            plots.append(
                (
                    fig_estimate_vs_target,
                    axs_estimate_vs_target,
                    ChartType.KALMAN_VS_HARMONIC_FIT,
                )
            )
        if (
            ChartType.ESTIMATES_INTERCEPT_COS_SIN in flags
            and flags[ChartType.ESTIMATES_INTERCEPT_COS_SIN]
        ):
            fig_intercept_cos_sin, axs_intercept_cos_sin = plt.subplots(
                1, 1, figsize=ASPECT_RATIO
            )
            plots.append(
                (
                    fig_intercept_cos_sin,
                    axs_intercept_cos_sin,
                    ChartType.ESTIMATES_INTERCEPT_COS_SIN,
                )
            )
        if (
            ChartType.RESIDUALS_OVER_TIME in flags
            and flags[ChartType.RESIDUALS_OVER_TIME]
        ):
            fig_residuals, axs_residuals = plt.subplots(1, 1, figsize=ASPECT_RATIO)
            plots.append((fig_residuals, axs_residuals, ChartType.RESIDUALS_OVER_TIME))

        if ChartType.AMPLITUDE in flags and flags[ChartType.AMPLITUDE]:
            fig_amplitude, axs_amplitude = plt.subplots(1, 1, figsize=ASPECT_RATIO)
            plots.append((fig_amplitude, axs_amplitude, ChartType.AMPLITUDE))

        if ChartType.BULC_PROBS in flags and flags[ChartType.BULC_PROBS]:
            fig_bulc_probs, axs_bulc_probs = plt.subplots(1, 1, figsize=ASPECT_RATIO)
            plots.append((fig_bulc_probs, axs_bulc_probs, ChartType.BULC_PROBS))

        if ChartType.KALMAN_VS_CCDC in flags and flags[ChartType.KALMAN_VS_CCDC]:
            fig_kalman_vs_ccdc, axs_kalman_vs_ccdc = plt.subplots(
                1, 1, figsize=ASPECT_RATIO
            )
            plots.append(
                (fig_kalman_vs_ccdc, axs_kalman_vs_ccdc, ChartType.KALMAN_VS_CCDC)
            )

        eeek_output = pd.read_csv(data_file_path)

        for fig, axes, graph_type in plots:
            if graph_type == ChartType.KALMAN_VS_HARMONIC_FIT:
                estimate(
                    axes,
                    eeek_output.copy(),
                    target_observations.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                    include_2022_fit=False,
                    include_2023_fit=False,
                )
            elif graph_type == ChartType.ESTIMATES_INTERCEPT_COS_SIN:
                intercept_cos_sin(
                    axes,
                    eeek_output.copy(),
                    target_observations.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                )
            elif graph_type == ChartType.RESIDUALS_OVER_TIME:
                residuals_over_time(
                    axes,
                    eeek_output.copy(),
                    target_observations.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                )
            elif graph_type == ChartType.AMPLITUDE:
                amplitude(
                    axes,
                    eeek_output.copy(),
                    target_observations.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                )
            elif graph_type == ChartType.BULC_PROBS:
                bulc_probs(
                    axes,
                    eeek_output.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                )
            elif graph_type == ChartType.KALMAN_VS_CCDC:
                estimate_vs_ccdc(
                    axes,
                    eeek_output.copy(),
                    point_index,
                    CHART_TITLES[graph_type],
                )

        for fig, axs, graph_type in plots:
            labels, handles = get_labels_and_handles(axs)
            fig.legend(handles, labels, loc="upper center", ncol=5)

            plt.tight_layout()
            save_chart(fig, point_index, CHART_TITLES[graph_type], output_directory)
            plt.close(fig)
