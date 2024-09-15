import pandas as pd
import numpy as np
import math

rmse = lambda x, y: math.sqrt(((x - y) ** 2).mean())


def calculate_rmse(actual_data_path, expected_data_path):
    actual_data = pd.read_csv(actual_data_path)
    expected_data = pd.read_csv(expected_data_path)

    merged_data = pd.merge(actual_data, expected_data, on="point")

    grouped_by_point = merged_data.groupby("point")

    rmses = []

    for group_name, group_data in grouped_by_point:
        intercept_rms = rmse(group_data["INTP"], group_data["intercept"])
        cos_rms = rmse(group_data["COS0"], group_data["cos"])
        sin_rms = rmse(group_data["SIN0"], group_data["sin"])
        sum_rms = intercept_rms + cos_rms + sin_rms

        rmses.append([group_name, intercept_rms, cos_rms, sin_rms, sum_rms])

    total_intercept_rms = rmse(merged_data["INTP"], merged_data["intercept"])
    total_cos_rms = rmse(merged_data["COS0"], merged_data["cos"])
    total_sin_rms = rmse(merged_data["SIN0"], merged_data["sin"])
    total_sum_rms = total_intercept_rms + total_cos_rms + total_sin_rms

    rmses.append(
        ["total", total_intercept_rms, total_cos_rms, total_sin_rms, total_sum_rms]
    )

    return rmses
