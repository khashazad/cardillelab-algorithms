import os
import shutil
from lib.constants import HARMONIC_FLAGS_LABEL, Harmonic, Kalman
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots
from lib.paths import (
    build_end_of_year_kalman_state_path,
    build_harmonic_trend_path,
    build_kalman_analysis_path,
    build_kalman_result_path,
)

run_directory = (
    "./tests/kalman/PNW_L7_L8_L9_swir_2017-2019_posthoc/11-24_22:13_slope_unimodal"
)

points = 7

for i in range(4, points):
    data = build_kalman_result_path(run_directory, i)
    harmonic_trend = build_harmonic_trend_path(run_directory, i)
    eoy_state = build_end_of_year_kalman_state_path(run_directory, i)

    analysis_directory = build_kalman_analysis_path(run_directory, i)

    harmonic_flags = {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.UNIMODAL.value: True,
        # Harmonic.BIMODAL.value: True,
        # Harmonic.TRIMODAL.value: False,
    }

    PLOT_OPTIONS = {
        # PlotType.KALMAN_VS_HARMONIC: {
        #     "harmonic_trend": harmonic_trend,
        #     "harmonic_flags": harmonic_flags,
        # },
        PlotType.KALMAN_RETROFITTED: {
            HARMONIC_FLAGS_LABEL: harmonic_flags,
            Kalman.EOY_STATE.value: eoy_state,
        },
        # PlotType.KALMAN_VS_CCDC_COEFS: {
        #     "title": "Kalman vs CCDC Coefficients",
        #     HARMONIC_FLAGS_LABEL: harmonic_flags,
        #     Kalman.EOY_STATE.value: eoy_state,
        # },
    }

    if os.path.exists(analysis_directory):
        shutil.rmtree(analysis_directory)

    generate_plots(data, analysis_directory, PLOT_OPTIONS, display=True)
