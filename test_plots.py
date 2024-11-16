import os
import shutil
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots

run_directory = "./tests/kalman/PNW_L7_L8_swir_2012_2013_2014_posthoc/11-15 20:02"

data = f"{run_directory}/result/eeek_0.csv"
harmonic_trend = f"{run_directory}/harmonic_coefficients/fitted_coefficients_0.csv"

analysis_directory = f"{run_directory}/analysis/0"


PLOT_OPTIONS = {
    PlotType.KALMAN_VS_HARMONIC: {
        "harmonic_trend": harmonic_trend,
    },
}
if os.path.exists(analysis_directory):
    shutil.rmtree(analysis_directory)

generate_plots(data, analysis_directory, PLOT_OPTIONS)
