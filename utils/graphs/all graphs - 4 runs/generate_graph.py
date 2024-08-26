import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import shutil
import os
import shutil


script_directory = os.path.dirname(os.path.abspath(__file__))

images_directory = f"{script_directory}/images"

if os.path.exists(images_directory):
    shutil.rmtree(images_directory)

data_files = ["v1.csv", "v2.csv", "v3.csv"]

target_observations = pd.read_csv(os.path.relpath(f"{script_directory}/data/observations.csv"))

points_count = len(target_observations['point'].unique())

flag = {
    "estimate": True,
    "final_2023_fit": True,
    "final_2022_fit": True,
    "intercept_cos_sin": True,
    "residuals": True
}

def get_subgraph_title(index): 
    if index == 0:
        return 'Default parameters'
    elif index == 1:
        return 'Default parameters increased by factor of 2'
    elif index == 2:
        return 'Default parameters increased by factor of 10'
    elif index == 3:
        return 'Far from default parameters'
    

def estimate(axs, actual, expected, point_index, file_index):
    axes = axs[file_index // 2, file_index % 2]

    actual["time"] = (pd.to_datetime(actual['date'], unit='ms')- pd.to_datetime('2016-01-01')).dt.total_seconds()  / (365.25 * 24 * 60 * 60)

    phi = 6.283 * actual["time"]
    phi_cos = phi.apply(math.cos)
    phi_sin = phi.apply(math.sin)

    actual['observation_estimate'] = expected["intercept"] + expected["cos"] * phi_cos + expected["sin"] * phi_sin

    actual['date'] = pd.to_datetime(actual['date'], unit='ms')
    
    filtered_data = actual[(actual['z'] != 0) & (actual['point'] == point_index)]

    axes.plot(filtered_data['date'], filtered_data['estimate'], label='Estimate - Optimized', linestyle='-', color='blue')
    axes.plot(filtered_data['date'], filtered_data['observation_estimate'], label='Target', linestyle='--', color='green')
    axes.scatter(filtered_data['date'], filtered_data['z'], label='Observed', s=10, color='red')

    if flag["final_2022_fit"]:
        filtered_data_2022 = filtered_data[filtered_data['date'].dt.year == 2022]
        intercept = filtered_data_2022["INTP"].iloc[-1]
        cos0 = filtered_data_2022["COS0"].iloc[-1]
        sin0 = filtered_data_2022["SIN0"].iloc[-1] 

        x = filtered_data["time"]

        A = np.sqrt(cos0**2 + sin0**2)
        phi = np.arctan2(sin0, cos0)

        y = intercept + A * np.cos(2 * np.pi * x - phi)

        axes.plot(filtered_data['date'], y, label="Final 2022 fit", color='orange')

    if flag["final_2023_fit"]:
        filtered_data_2023 = filtered_data[filtered_data['date'].dt.year == 2023]
        intercept = filtered_data_2023["INTP"].iloc[-1]
        cos0 = filtered_data_2023["COS0"].iloc[-1]
        sin0 = filtered_data_2023["SIN0"].iloc[-1] 

        x = filtered_data["time"]

        A = np.sqrt(cos0**2 + sin0**2)
        phi = np.arctan2(sin0, cos0)

        y = intercept + A * np.cos(2 * np.pi * x - phi)

        axes.plot(filtered_data['date'], y, label="Final 2023 fit", color='purple')

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    axes.set_title(get_subgraph_title(file_index))
    # axes.legend()

def intercept_cos_sin(axs, actual, expected, point_index, file_index):

    axes = axs[file_index // 2, file_index % 2]
    actual['target_intercept'] = expected["intercept"]
    actual['target_cos'] = expected["cos"]
    actual['target_sin'] = expected["sin"]

    actual['date'] = pd.to_datetime(actual['date'], unit='ms')
    
    filtered_data = actual[(actual['z'] != 0) & (actual['point'] == point_index)]

    dates = filtered_data['date']
    intp = filtered_data['INTP']
    cos = filtered_data['COS0']
    sin = filtered_data['SIN0']
    target_intercept = filtered_data['target_intercept']
    target_cos = filtered_data['target_cos']
    target_sin = filtered_data['target_sin']

    axes.plot(dates, intp, label='intercept', linestyle='-', color='blue')
    axes.plot(dates, target_intercept, label='target intercept', linestyle='--', color='blue')

    axes.plot(dates, cos, label='cos', linestyle='-', color='green')
    axes.plot(dates, target_cos, label='target cos', linestyle='--', color='green')

    axes.plot(dates, sin, label='sin', linestyle='-', color='red')
    axes.plot(dates, target_sin, label='target sin', linestyle='--', color='red')

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    axes.set_title(get_subgraph_title(file_index))
    # axes.legend()

def residuals_over_time(axs, actual, expected, point_index, file_index):

    axes = axs[file_index // 2, file_index % 2]

    actual['intercept_residual'] = actual['INTP'] - expected["intercept"]
    actual['cos_residual'] = actual['COS0'] - expected["cos"]
    actual['sin_residual'] = actual['SIN0'] - expected["sin"]

    actual['date'] = pd.to_datetime(actual['date'], unit='ms')
    
    filtered_data = actual[(actual['z'] != 0) & (actual['point'] == point_index)]

    intercept_residual = filtered_data['intercept_residual']
    cos_residual = filtered_data['cos_residual']
    sin_residual = filtered_data['sin_residual']
    dates = filtered_data['date']

    axes.scatter(dates, intercept_residual, label='intercept', linestyle='-', color='blue', s=10)
    axes.scatter(dates, cos_residual, label='cos', linestyle='-', color='green', s=10)
    axes.scatter(dates, sin_residual, label='sin', linestyle='-', color='red', s=10)

    axes.axhline(y=0, color='black', linestyle='--')

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    axes.set_title(get_subgraph_title(file_index))
    # axes.legend()

def save_graph(fig, point_index, name):
    point_directory = f"{script_directory}/images/points/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    fig.savefig(f"{point_directory}/{name}")

    directory = f"{script_directory}/images/{name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fig.savefig(f"{directory}/{point_index}")


def get_unique_labels_and_handles(axs):
    handles, labels = axs.get_legend_handles_labels()
    unique_labels = set()
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.add(label)

    return unique_labels, unique_handles

# estimate vs target
for point_index in range(points_count):
    fig_estimate_vs_target, axs_estimate_vs_target = plt.subplots(2, 2, figsize=(12, 8))
    fig_intercept_cos_sin, axs_intercept_cos_sin = plt.subplots(2, 2, figsize=(12, 8))
    fig_residuals, axs_residuals = plt.subplots(2, 2, figsize=(12, 8))

    plots = [
        (fig_estimate_vs_target, axs_estimate_vs_target),
        (fig_intercept_cos_sin, axs_intercept_cos_sin),
        (fig_residuals, axs_residuals)
    ]

    for file_index, data_file in enumerate(data_files):
        eeek_output = pd.read_csv(os.path.relpath(f"{script_directory}/data/{data_file}"))

        estimate(axs_estimate_vs_target, eeek_output.copy(), target_observations.copy(), point_index, file_index)

        intercept_cos_sin(axs_intercept_cos_sin, eeek_output.copy(), target_observations.copy(), point_index, file_index)

        residuals_over_time(axs_residuals, eeek_output.copy(), target_observations.copy(), point_index, file_index)

        
    for fig, axs in plots:
        labels, handles = get_unique_labels_and_handles(axs[0, 0])
        fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout()

    if flag["estimate"]:
        save_graph(fig_estimate_vs_target, point_index, "estimate vs target")
    if flag["intercept_cos_sin"]:
        save_graph(fig_intercept_cos_sin, point_index, "intercept cos sin")
    if flag["residuals"]:
        save_graph(fig_residuals, point_index, "residuals over time")
