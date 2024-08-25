import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt


all_data = pd.read_csv(os.path.relpath("./graphs/curve with final coefficients/data.csv"))

all_data["time"] = (pd.to_datetime(all_data['date'], unit='ms') - pd.to_datetime('2016-01-01')).dt.total_seconds() / (365.25 * 24 * 60 * 60)

# print(all_data["time"].to_string())

# all_data['date'] = pd.to_datetime(all_data['date'], unit='ms')

# filtered_data = all_data

filtered_data = all_data[all_data['z'] != 0]

grouped_data = filtered_data.groupby(filtered_data["point"])

plt.figure(figsize=(10, 6))

for group_name, data in grouped_data:

    dates = data['date']

    intercept = data["INTP"].iloc[-1]
    cos0 = data["COS0"].iloc[-1]
    sin0 = data["SIN0"].iloc[-1]

    print(intercept, cos0, sin0)    

    # Define the parameters
    beta0 = intercept
    A = np.sqrt(cos0**2 + sin0**2)
    phi = np.arctan2(sin0, cos0)

    # print(beta0, A, phi)

    # Generate the modified curve
    x = data["time"]
    y = beta0 + A * np.cos(2 * np.pi * x - phi)

    # print(x.to_string())

    # Plot the measurements value from the z column
    z = data["z"]
    # plt.plot(x, z, label="Measurements")

    # Plot the modified curve
    plt.plot(dates, y, label="Modified Curve")
    plt.scatter(dates, z, label="Measurements")


    # Add labels and legend
    plt.xlabel("Point")
    plt.ylabel("Value")
    plt.legend()

    # exit()
    # Show the plot
    plt.show()