import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


script_directory = os.path.dirname(os.path.abspath(__file__))

data_files = ["v1.csv", "v2.csv", "v3.csv", "v4.csv"]

directory = "all graphs - 4 runs"

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

# estimate vs target
for point_index in range(points_count):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/observations.csv"))

        year_difference = (pd.to_datetime(data['date'], unit='ms')- pd.to_datetime('2016-01-01')).dt.total_seconds()  / (365.25 * 24 * 60 * 60)

        phi = 6.283 * year_difference
        phi_cos = phi.apply(math.cos)
        phi_sin = phi.apply(math.sin)

        data['observation_estimate'] = observations["intercept"] + observations["cos"] * phi_cos + observations["sin"] * phi_sin

        data['date'] = pd.to_datetime(data['date'], unit='ms')
        
        filtered_data = data[data['z'] != 0]

        filtered_data = filtered_data[filtered_data['point'] == point_index]

        grouped_data = filtered_data.groupby(filtered_data["point"])

        dates = filtered_data['date']
        estimate = filtered_data['estimate']
        z = filtered_data['z']
        intp = filtered_data['INTP']

        ax.plot(dates, estimate, label='Estimate - Optimized', linestyle='-')
        ax.plot(dates, filtered_data['observation_estimate'], label='Target', linestyle='--', color='green')
        ax.scatter(dates, z, label='Observed', s=10, color='red')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        ax.set_title(get_subgraph_title(index))
        # ax.legend()

    plt.tight_layout()

    point_directory = f"./graphs/{directory}/images/estimate vs target/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    plt.savefig(f"{point_directory}/estimate_vs_target_{point_index}")
    # plt.show()

# estimate of intercept cos sin
for point_index in range(points_count):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/observations.csv"))

        data['target_intercept'] = observations["intercept"]
        data['target_cos'] = observations["cos"]
        data['target_sin'] = observations["sin"]

        data['date'] = pd.to_datetime(data['date'], unit='ms')
        
        filtered_data = data[data['z'] != 0]

        filtered_data = filtered_data[filtered_data['point'] == point_index]

        grouped_data = filtered_data.groupby(filtered_data["point"])

        dates = filtered_data['date']
        intp = filtered_data['INTP']
        cos = filtered_data['COS0']
        sin = filtered_data['SIN0']
        target_intercept = filtered_data['target_intercept']
        target_cos = filtered_data['target_cos']
        target_sin = filtered_data['target_sin']

        ax.plot(dates, intp, label='intercept', linestyle='-', color='blue')
        ax.plot(dates, target_intercept, label='target intercept', linestyle='--', color='blue')

        ax.plot(dates, cos, label='cos', linestyle='-', color='green')
        ax.plot(dates, target_cos, label='target cos', linestyle='--', color='green')

        ax.plot(dates, sin, label='sin', linestyle='-', color='red')
        ax.plot(dates, target_sin, label='target sin', linestyle='--', color='red')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        ax.set_title(get_subgraph_title(index))
        # ax.legend()

    plt.tight_layout()

    point_directory = f"./graphs/{directory}/images/estimate vs target/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    plt.savefig(f"{point_directory}/estimate_of_intp_cos_sin_{point_index}")

# amplitude
for point_index in range(points_count):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/observations.csv"))
        
        data['target_amplitude'] = np.sqrt(observations["cos"]**2 + observations["sin"]**2)

        data['date'] = pd.to_datetime(data['date'], unit='ms')

        filtered_data = data[data['z'] != 0]

        filtered_data = filtered_data[filtered_data['point'] == point_index]

        dates = filtered_data['date']
        estimate = filtered_data['estimate']
        z = filtered_data['z']
        intp = filtered_data['INTP']

        cos = filtered_data["COS0"]
        sin = filtered_data["SIN0"]

        amplitude = np.sqrt(cos**2 + sin**2)
        
        ax.plot(dates, amplitude, label="Amplitude", linestyle='-', color='blue')
        ax.plot(dates, filtered_data['target_amplitude'], label="Target Amplitude", linestyle='--', color='green')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        ax.set_title(get_subgraph_title(index))
        # ax.legend()

    plt.tight_layout()
    point_directory = f"./graphs/{directory}/images/estimate vs target/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    plt.savefig(f"{point_directory}/amplitude_{point_index}")
    # plt.show()

# final coefficients vs observed
for point_index in range(points_count):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/{directory}/data/observations.csv"))
        
        data['target_amplitude'] = np.sqrt(observations["cos"]**2 + observations["sin"]**2)

        data['date'] = pd.to_datetime(data['date'], unit='ms')

        data["time"] = (pd.to_datetime(data['date'], unit='ms') - pd.to_datetime('2016-01-01')).dt.total_seconds() / (365.25 * 24 * 60 * 60)

        filtered_data = filtered_data[filtered_data['point'] == point_index]

        # print(len(filtered_data))

        # intercept_2022 = filtered_data["INTP"].iloc[24]
        # cos0_2022 = filtered_data["COS0"].iloc[24]
        # sin0_2022 = filtered_data["SIN0"].iloc[24]

        filtered_data = filtered_data[filtered_data['z'] != 0]

        dates = filtered_data['date']

        intercept = filtered_data["INTP"].iloc[-1]
        cos0 = filtered_data["COS0"].iloc[-1]
        sin0 = filtered_data["SIN0"].iloc[-1]  

        x = filtered_data["time"]

        # A_2022 = np.sqrt(cos0_2022**2 + sin0_2022**2)   
        # phi_2022 = np.arctan2(sin0_2022, cos0_2022)

        # y_2022 = intercept_2022 + A_2022 * np.cos(2 * np.pi * x - phi_2022)

        A = np.sqrt(cos0**2 + sin0**2)
        phi = np.arctan2(sin0, cos0)

        y = intercept + A * np.cos(2 * np.pi * x - phi)

        z = filtered_data["z"]

        # Plot the modified curve
        ax.plot(dates, y, label="Fit with final coefficients", color='blue')
        # ax.plot(dates, y_2022, label="Fit with final 2022 coefficients", color='green')
        ax.scatter(dates, z, label="Measurements", color='red')


        # Add labels and legend 
        ax.set_title(get_subgraph_title(index))
        # ax.legend()

    plt.tight_layout()
    plt.legend()
    point_directory = f"./graphs/{directory}/images/estimate vs target/{point_index}"
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)
    plt.savefig(f"{point_directory}/fit_with_final_coefficients_{point_index}")