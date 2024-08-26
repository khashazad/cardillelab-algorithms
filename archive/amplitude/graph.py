import pandas as pd
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data_files = ["v1.csv", "v2.csv", "v3.csv", "v4.csv"]


for point_index in range(6):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for index, data_file in enumerate(data_files):
        ax = axs[index // 2, index % 2]

        data = pd.read_csv(os.path.relpath(f"./graphs/amplitude/data/{data_file}"))
        observations = pd.read_csv(os.path.relpath(f"./graphs/amplitude/data/observations.csv"))
        
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
    plt.savefig(f"./graphs/amplitude/images/{point_index}")
    # plt.show()
