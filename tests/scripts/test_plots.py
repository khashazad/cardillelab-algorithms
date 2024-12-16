import os
import shutil
from lib.constants import (
    FORWARD_TREND_LABEL,
    HARMONIC_FLAGS_LABEL,
    HARMONIC_TREND_LABEL,
    Harmonic,
    Kalman,
)
from lib.utils.visualization.constant import PlotType
from lib.utils.visualization.plot_generator import generate_plots
from lib.paths import (
    build_end_of_year_kalman_state_path,
    build_harmonic_trend_path,
    build_kalman_analysis_path,
    build_kalman_result_path,
)

run_directory = (
    "./tests/kalman/Randonia_L7_L8_L9_swir_2015-2023_randonia_2/12-15_20:45_unimodal"
)

points = int(run_directory.split("/")[-2].split("_")[-1])

for i in range(0, points):
    data = build_kalman_result_path(run_directory, i)
    harmonic_trend = build_harmonic_trend_path(run_directory, i)
    eoy_state = build_end_of_year_kalman_state_path(run_directory, i)

    analysis_directory = build_kalman_analysis_path(run_directory, i)

    harmonic_flags = {
        Harmonic.INTERCEPT.value: True,
        # Harmonic.SLOPE.value: True,
        Harmonic.UNIMODAL.value: True,
        # Harmonic.BIMODAL.value: True,
        # Harmonic.TRIMODAL.value: False,
    }

    PLOT_OPTIONS = {
        PlotType.KALMAN_VS_HARMONIC: {
            "title": f"Kalman vs Harmonic Trend",
            HARMONIC_TREND_LABEL: build_harmonic_trend_path(run_directory, i),
            HARMONIC_FLAGS_LABEL: harmonic_flags,
        },
        PlotType.KALMAN_FIT: {
            "title": "Kalman Fit",
        },
        PlotType.KALMAN_VS_CCDC: {
            "title": "Kalman vs CCDC",
        },
        PlotType.KALMAN_VS_CCDC_COEFS: {
            "title": "Kalman vs CCDC Coefficients",
            HARMONIC_FLAGS_LABEL: harmonic_flags,
        },
        PlotType.KALMAN_RETROFITTED: {
            HARMONIC_FLAGS_LABEL: harmonic_flags,
            Kalman.EOY_STATE.value: build_end_of_year_kalman_state_path(
                run_directory, i
            ),
            FORWARD_TREND_LABEL: True,
            HARMONIC_TREND_LABEL: build_harmonic_trend_path(run_directory, i),
        },
        PlotType.KALMAN_YEARLY_FIT: {
            HARMONIC_FLAGS_LABEL: harmonic_flags,
            "title": "Kalman Yearly Fit",
            Kalman.EOY_STATE.value: build_end_of_year_kalman_state_path(
                run_directory, i
            ),
        },
    }

    if os.path.exists(analysis_directory):
        shutil.rmtree(analysis_directory)

    generate_plots(data, analysis_directory, PLOT_OPTIONS, display=False)
