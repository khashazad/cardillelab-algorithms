import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv(os.path.relpath("./graph/estimate of intp cos sin/data/pest_output.csv"))

filtered_data = data[data['z'] != 0]

data['date'] = pd.to_datetime(data['date'], unit='ms')

grouped_data = filtered_data.groupby(filtered_data["point"])

plt.figure(figsize=(10, 6))

for group_name, group_data in grouped_data:

    dates = group_data['date']
    dates = pd.to_datetime(group_data['date'], unit='ms')


    intp = group_data['INTP']
    cos = group_data['COS0']
    sin = group_data['SIN0']

    plt.plot(dates, intp, label='Intercept', linestyle='-')
    plt.plot(dates, cos, label='COS', linestyle='-')
    plt.plot(dates, sin, label='SIN', linestyle='-')
    # plt.scatter(dates, z, label='Observed', s=10)

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # plt.title(f'Kalman Estimate - Point {group_name}')
    plt.legend()

    plt.savefig(f"./graph/estimate of intp cos sin/graphs/graph_group_{group_name}")
    plt.show()
