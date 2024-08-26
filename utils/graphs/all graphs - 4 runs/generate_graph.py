import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


script_directory = os.path.dirname(os.path.abspath(__file__))

data_files = ["v1.csv", "v2.csv"]

points_count = 9

def get_subgraph_title(index): 
    if index == 0:
        return 'Default parameters'
    elif index == 1:
        return 'Default parameters increased by factor of 2'
    elif index == 2:
        return 'Default parameters increased by factor of 10'
    elif index == 3:
        return 'Far from default parameters'
    

def estimate(axes, eeek_output, target_observations, point_index, file_index):
    year_difference = (pd.to_datetime(eeek_output['date'], unit='ms')- pd.to_datetime('2016-01-01')).dt.total_seconds()  / (365.25 * 24 * 60 * 60)

    phi = 6.283 * year_difference
    phi_cos = phi.apply(math.cos)
    phi_sin = phi.apply(math.sin)

    eeek_output['observation_estimate'] = target_observations["intercept"] + target_observations["cos"] * phi_cos + target_observations["sin"] * phi_sin

    eeek_output['date'] = pd.to_datetime(eeek_output['date'], unit='ms')
    
    filtered_data = eeek_output[(eeek_output['z'] != 0) & (eeek_output['point'] == point_index)]

    axes.plot(filtered_data['date'], filtered_data['estimate'], label='Estimate - Optimized', linestyle='-')
    axes.plot(filtered_data['date'], filtered_data['observation_estimate'], label='Target', linestyle='--', color='green')
    axes.scatter(filtered_data['date'], filtered_data['z'], label='Observed', s=10, color='red')

    axes.xaxis.set_major_locator(mdates.AutoDateLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    axes.set_title(get_subgraph_title(file_index))
    axes.legend()

def intercept_cos_sin(axes, eeek_output, target_observations, point_index, file_index):

    eeek_output['target_intercept'] = target_observations["intercept"]
    eeek_output['target_cos'] = target_observations["cos"]
    eeek_output['target_sin'] = target_observations["sin"]

    eeek_output['date'] = pd.to_datetime(eeek_output['date'], unit='ms')
    
    filtered_data = eeek_output[(eeek_output['z'] != 0) & (eeek_output['point'] == point_index)]

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
    axes.legend()

def save_graph(fig, point_index, name):
    point_directory = f"{script_directory}/images/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    fig.savefig(f"{point_directory}/{name}")

# estimate vs target
for point_index in range(points_count):
    fig_estimate_vs_target, axs_estimate_vs_target = plt.subplots(2, 2, figsize=(12, 8))
    fig_intercept_cos_sin, axs_intercept_cos_sin = plt.subplots(2, 2, figsize=(12, 8))

    for file_index, data_file in enumerate(data_files):
        eeek_output = pd.read_csv(os.path.relpath(f"{script_directory}/data/{data_file}"))
        target_observations = pd.read_csv(os.path.relpath(f"{script_directory}/data/observations.csv"))

        ax_estimate_vs_target = axs_estimate_vs_target[file_index // 2, file_index % 2]
        estimate(ax_estimate_vs_target, eeek_output.copy(), target_observations.copy(), point_index, file_index)

        ax_intercept_cos_sin = axs_intercept_cos_sin[file_index // 2, file_index % 2]
        intercept_cos_sin(ax_intercept_cos_sin, eeek_output.copy(), target_observations.copy(), point_index, file_index)

    plt.tight_layout()

    save_graph(fig_estimate_vs_target, point_index, "estimate vs target")
    save_graph(fig_intercept_cos_sin, point_index, "intercept cos sin")