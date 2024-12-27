""" Paths used throughout project. """

from datetime import datetime
import os

from lib.constants import Harmonic

# directories
RESULTS_DIRECTORY = "result"
ANALYSIS_DIRECTORY = "analysis"

# subdirectories
HARMONIC_TREND_SUBDIRECTORY = "harmonic_trend"
KALMAN_STATE_SUBDIRECTORY = "kalman_state"
KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY = "kalman_end_of_year_state"
CCDC_SEGMENTS_SUBDIRECTORY = "ccdc_segments"

# file prefixes
POINTS_FILE_PREFIX = "points"
KALMAN_OUTPUT_FILE_PREFIX = "state_point"
END_OF_YEAR_KALMAN_STATE_FILE_PREFIX = "eoy_state_point"
HARMONIC_TREND_COEFS_FILE_PREFIX = "harmonic_trend_point"
CCDC_SEGMENTS_FILE_PREFIX = "ccdc_segments"


def kalman_result_directory(run_directory: str) -> str:
    return os.path.join(run_directory, RESULTS_DIRECTORY)


def kalman_analysis_directory(run_directory: str) -> str:
    return os.path.join(run_directory, ANALYSIS_DIRECTORY)


def build_kalman_result_path(run_directory: str, index: int) -> str:
    return os.path.join(
        kalman_result_directory(run_directory),
        KALMAN_STATE_SUBDIRECTORY,
        f"{KALMAN_OUTPUT_FILE_PREFIX}_{index}.csv",
    )


def build_harmonic_trend_path(run_directory: str, index: int) -> str:
    return os.path.join(
        kalman_result_directory(run_directory),
        HARMONIC_TREND_SUBDIRECTORY,
        f"{HARMONIC_TREND_COEFS_FILE_PREFIX}_{index}.csv",
    )


def build_kalman_analysis_path(run_directory: str, index: int) -> str:
    return os.path.join(kalman_analysis_directory(run_directory), f"{index}")


def build_end_of_year_kalman_state_path(run_directory: str, index: int) -> str:
    return os.path.join(
        kalman_result_directory(run_directory),
        KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY,
        f"{END_OF_YEAR_KALMAN_STATE_FILE_PREFIX}_{index}.csv",
    )


def build_ccdc_segments_path(run_directory: str, index: int) -> str:
    return os.path.join(
        kalman_result_directory(run_directory),
        CCDC_SEGMENTS_SUBDIRECTORY,
        f"{CCDC_SEGMENTS_FILE_PREFIX}_{index}.json",
    )


def build_points_path(run_directory: str) -> str:
    return os.path.join(
        run_directory,
        f"{POINTS_FILE_PREFIX}.json",
    )


def build_kalman_run_directory(
    script_directory: str, tag: str, harmonic_flags: dict, run_id: str = None
) -> str:
    
    flags_prefix = ""

    if harmonic_flags is not None:
        include_slope = harmonic_flags.get(Harmonic.SLOPE.value, False)
        bimodal = harmonic_flags.get(Harmonic.BIMODAL.value, False)
        trimodal = harmonic_flags.get(Harmonic.TRIMODAL.value, False)

        if include_slope:
            flags_prefix += "_slope"
        if trimodal:
            flags_prefix += "_trimodal"
        elif bimodal:
            flags_prefix += "_bimodal"
        else:
            flags_prefix += "_unimodal"

    run_id_prefix = f"{run_id}_" if run_id and run_id != "" else ""
    date_time_str = datetime.now().strftime("%m-%d_%H:%M")

    return f"{script_directory}/tests/kalman/{tag}/{run_id_prefix}{date_time_str}{flags_prefix}/"


def get_kalman_parameters_path(script_directory: str, harmonic_flags: dict) -> str:
    include_slope = harmonic_flags.get(Harmonic.SLOPE.value, False)
    bimodal = harmonic_flags.get(Harmonic.BIMODAL.value, False)
    trimodal = harmonic_flags.get(Harmonic.TRIMODAL.value, False)

    postfix = ""
    if include_slope:
        postfix += "_slope"
    if trimodal:
        postfix += "_trimodal"
    elif bimodal:
        postfix += "_bimodal"
    else:
        postfix += "_unimodal"

    return f"{script_directory}/kalman/kalman_parameters{postfix}.json"
