import pandas as pd
import os
import matplotlib.pyplot as plt
from lib.constants import DATE_LABEL
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
from lib.utils.visualization.plots.kalman_vs_ccdc_coefs_plot import (
    kalman_vs_ccdc_coefs_plot,
)
from lib.utils.visualization.plots.kalman_retrofitted import kalman_retrofitted_plot
from lib.utils.visualization.plots.kalman_yearly_fit import kalman_yearly_fit_plot


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


def save_chart(fig, name, output_directory, legend):
    os.makedirs(output_directory, exist_ok=True)
    fig.savefig(
        f"{output_directory}/{name}.png",
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )

def generate_yearly_kalman_fit_plots(data, output_path, options, display=False):
    unique_years = (
        pd.to_datetime(data[DATE_LABEL]).dt.year.unique().tolist()
    )

    for year in unique_years:
        fig, axs = plt.subplots(figsize=ASPECT_RATIO)

        kalman_yearly_fit_plot(
            axs,
            data.copy(),
            options[PlotType.KALMAN_YEARLY_FIT],
            year,
        )

        labels, handles = get_labels_and_handles(axs)

        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=6 if len(labels) > 6 else len(labels),
        )

        save_chart(
            fig,
            f"{year}",
            f"{output_path}/{PlotType.KALMAN_YEARLY_FIT.value}",
            legend=legend,
        )

def generate_plots(data, output_path, options, display=False):
    # create output directory
    os.makedirs(output_path, exist_ok=True)

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
        elif plot_type == PlotType.KALMAN_VS_CCDC_COEFS:
            kalman_vs_ccdc_coefs_plot(
                axes,
                kalman_output.copy(),
                options[plot_type],
            )
        elif plot_type == PlotType.KALMAN_RETROFITTED:
            kalman_retrofitted_plot(
                axes,
                kalman_output.copy(),
                options[plot_type],
            )

    if options.get(PlotType.KALMAN_YEARLY_FIT, False):
        generate_yearly_kalman_fit_plots(kalman_output, output_path, options, display)

    for fig, axs, plot_type in plots:
        labels, handles = get_labels_and_handles(axs)
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=6 if len(labels) > 6 else len(labels),
        )

        plt.tight_layout()
        save_chart(fig, plot_type.value, output_path, legend=legend)

        if display:
            plt.show()

        plt.close(fig)
