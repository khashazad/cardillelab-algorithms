""" Paths used throughout project. """

import os


# directories
RESULTS_DIRECTORY = "result"
ANALYSIS_DIRECTORY = "analysis"

# subdirectories
HARMONIC_TREND_SUBDIRECTORY = "harmonic_trend"
KALMAN_STATE_SUBDIRECTORY = "kalman_state"
KALMAN_END_OF_YEAR_STATE_SUBDIRECTORY = "kalman_end_of_year_state"

# file prefixes
POINTS_FILE_PREFIX = "points"
KALMAN_OUTPUT_FILE_PREFIX = "state_point"
END_OF_YEAR_KALMAN_STATE_FILE_PREFIX = "eoy_state_point"
HARMONIC_TREND_COEFS_FILE_PREFIX = "harmonic_trend_point"


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


def build_points_path(run_directory: str) -> str:
    return os.path.join(
        run_directory,
        f"{POINTS_FILE_PREFIX}.csv",
    )
