import os
import shutil
from lib.constants import Harmonic
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots

run_directory = "./tests/kalman/PNW_L7_L8_swir_2015_2016_posthoc/11-15 23:06"

data = f"{run_directory}/result/eeek_0.csv"
harmonic_trend = f"{run_directory}/harmonic_coefficients/fitted_coefficients_0.csv"

analysis_directory = f"{run_directory}/analysis/0"


PLOT_OPTIONS = {
    PlotType.KALMAN_VS_HARMONIC: {
        "harmonic_trend": harmonic_trend,
        "harmonic_flags": {
            Harmonic.INTERCEPT.value: True,
            Harmonic.SLOPE.value: False,
            Harmonic.UNIMODAL.value: True,
            Harmonic.BIMODAL.value: True,
            Harmonic.TRIMODAL.value: False,
        },
    },
}
if os.path.exists(analysis_directory):
    shutil.rmtree(analysis_directory)

generate_plots(data, analysis_directory, PLOT_OPTIONS)
