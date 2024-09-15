import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


script_directory = os.path.dirname(os.path.abspath(__file__))


    
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

def save_graph(fig, name):
    fig.savefig(f"{script_directory}/images/{name}")

eeek_output = pd.read_csv(os.path.relpath(f"{script_directory}/data/pest_output.csv"))

filtered_data = eeek_output[(eeek_output['z'] != 0)]

data_grouped_by_point = filtered_data.groupby(filtered_data["point"])

points_count = len(data_grouped_by_point)

subplots = []
print(points_count)
for i in range(points_count // 4 + 1):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    subplots.append((fig, axs))

print(len(subplots))

for i, (point, data) in enumerate(data_grouped_by_point):
    ax = subplots[i // 4][1][i % 4 // 2, i % 2]

    dates= pd.to_datetime(data['date'], unit='ms')

    ax.scatter(dates, data['z'], label='z', linestyle='-', color='blue')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    ax.set_title(f"Point {point}")
    ax.legend()


for i, (fig, axes) in enumerate(subplots):
    fig.tight_layout()
    save_graph(fig, f"{i}.png")



    