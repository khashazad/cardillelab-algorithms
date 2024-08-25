import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data_files = ["v1.csv", "v2.csv", "v3.csv", "v4.csv"]

for point_index in range(6):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/estimate of intp cos sin - 4 runs/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/estimate of intp cos sin - 4 runs/data/observations.csv"))

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

        if index == 0:
            ax.set_title('Starting from default parameters')
        elif index == 1:
            ax.set_title('Starting from default parameters increased by factor of 2')
        elif index == 2:
            ax.set_title('Starting from default parameters increased by factor of 10')
        elif index == 3:
            ax.set_title('Starting from default parameters')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"./graphs/estimate of intp cos sin - 4 runs/images/{point_index}")
    # plt.show()
