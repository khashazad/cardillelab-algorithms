import pandas as pd
import os
import matplotlib.pyplot as plt
from lib.utils.visualization.constant import (
    ASPECT_RATIO,
    PLOT_TYPES,
    PlotType,
)
from lib.utils.visualization.plots.kalman_vs_harmonic_trend_plot import (
    kalman_estimate_vs_harmonic_trend_plot,
)
from lib.utils.visualization.plots.kalman_fit_plot import kalman_fit_plot
from lib.utils.visualization.plots.kalman_vs_ccdc_plot import kalman_vs_ccdc_plot


def save_chart(fig, name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    fig.savefig(f"{output_directory}/{name}.png")


def get_labels_and_handles(axs):
    handles, labels = axs.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    return unique_labels, unique_handles


def get_labels_and_handles(axs):
    handles, labels = axs.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    return unique_labels, unique_handles


def save_chart(fig, name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    fig.savefig(f"{output_directory}/{name}.png")


def generate_plots(data, output_directory, options):
    # create output directory
    os.makedirs(output_directory, exist_ok=True)

    plots = []

    for plot_type in PLOT_TYPES:
        if options.get(plot_type, None):
            fig, ax = plt.subplots(figsize=ASPECT_RATIO)
            plots.append((fig, ax, plot_type))

    kalman_output = pd.read_csv(data)

    for fig, axes, plot_type in plots:
        if plot_type == PlotType.KALMAN_VS_HARMONIC:
            kalman_estimate_vs_harmonic_trend_plot(
                axes,
                kalman_output.copy(),
                options[plot_type],
            )
        elif plot_type == PlotType.KALMAN_FIT:
            kalman_fit_plot(
                axes,
                kalman_output.copy(),
                options[plot_type],
            )
        elif plot_type == PlotType.KALMAN_VS_CCDC:
            kalman_vs_ccdc_plot(
                axes,
                kalman_output.copy(),
                options[plot_type],
            )

    for fig, axs, plot_type in plots:
        labels, handles = get_labels_and_handles(axs)
        fig.legend(handles, labels, loc="upper center", ncol=5)

        plt.tight_layout()
        save_chart(fig, plot_type.value, output_directory)
        plt.close(fig)
