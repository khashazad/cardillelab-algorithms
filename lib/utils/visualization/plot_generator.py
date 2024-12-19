import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def create_image_grids(image_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    image_files = [
        f for f in os.listdir(image_directory) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    image_files.sort()

    for i in range(0, len(image_files), 4):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        axs = axs.flatten()

        for ax, j in zip(axs, range(i, min(i + 4, len(image_files)))):
            img_path = os.path.join(image_directory, image_files[j])
            img = mpimg.imread(img_path)
            img = img[:-100, :, :]
            ax.imshow(img)
            ax.axis("off")

        output_path = os.path.join(output_directory, f"grid_{i // 4 + 1}.png")

        plt.tight_layout(pad=0)
        plt.savefig(output_path)
        plt.close(fig)


def generate_yearly_kalman_fit_plots(data, output_path, options, display=False):
    unique_years = pd.to_datetime(data[DATE_LABEL]).dt.year.unique().tolist()

    figures = []

    for year in unique_years:
        fig, axs = plt.subplots(figsize=ASPECT_RATIO)

        kalman_yearly_fit_plot(
            axs,
            data.copy(),
            options[PlotType.KALMAN_YEARLY_FIT],
            year,
        )

        figures.append((fig, axs))

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

        plt.close(fig)

    create_image_grids(
        f"{output_path}/{PlotType.KALMAN_YEARLY_FIT.value}",
        f"{output_path}/{PlotType.KALMAN_YEARLY_FIT.value}/4x4_grids",
    )


def generate_plots(data, output_path, options, display=False):
    # create output directory
    os.makedirs(output_path, exist_ok=True)

    plots = []

    for plot_type in PLOT_TYPES:
        if options.get(plot_type, None) and plot_type != PlotType.KALMAN_YEARLY_FIT:
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
