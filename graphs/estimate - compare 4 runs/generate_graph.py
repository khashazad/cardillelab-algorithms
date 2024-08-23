import pandas as pd
import os
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data_files = ["v1.csv", "v2.csv", "v3.csv", "v4.csv"]


for point_index in range(9):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/estimate - compare 4 runs/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/estimate - compare 4 runs/data/observations.csv"))

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

        if index == 0:
            ax.set_title('Starting from default parameters')
        elif index == 1:
            ax.set_title('Starting from default parameters increased by factor of 10')
        elif index == 2:
            ax.set_title('Starting from default parameters increased by factor of 100')
        elif index == 3:
            ax.set_title('Starting from default parameters')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"./graphs/estimate - compare 4 runs/images/{point_index}")
    # plt.show()
