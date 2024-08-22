import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

target_final_intercept = 1

data = pd.read_csv(os.path.relpath("./graph/estimate vs observed/data/v1.csv"))
data_v2 = pd.read_csv(os.path.relpath("./graph/estimate vs observed/data/v2.csv"))
data_original = pd.read_csv(os.path.relpath("./graph/estimate vs observed/data/original.csv"))

data['estimate_original'] = data_original['estimate']
data['estimate_v2'] = data_v2['estimate']
data['INTP_v2'] = data_v2['INTP']
data['date'] = pd.to_datetime(data['date'], unit='ms')

filtered_data = data[data['z'] != 0]

grouped_data = filtered_data.groupby(filtered_data["point"])

plt.figure(figsize=(10, 6))

for group_name, group_data in grouped_data:
    dates = group_data['date']
    estimate = group_data['estimate']
    estimate_v2 = group_data['estimate_v2']
    estimate_original = group_data['estimate_original']
    z = group_data['z']
    intp = group_data['INTP']
    intp_v2 = group_data['INTP_v2']  

    plt.plot(dates, estimate, label='Estimate - Optimized', linestyle='-')
    plt.plot(dates, estimate_v2, label='Estimate - Optimized - v2', linestyle='-')
    plt.plot(dates, estimate_original, label='Estimate - Original', linestyle='solid')
    plt.scatter(dates, z, label='Observed', s=10)
    # plt.plot(dates, [intp.iloc[-1]] * len(dates), label='Final Intercept', linestyle='--')
    # plt.plot(dates, [intp_v2.iloc[-1]] * len(dates), label='Final Intercept - v2', linestyle='--')
    # plt.plot(dates, [target_final_intercept] * len(dates), label='Target Intercept', linestyle='--')

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.title(f'Kalman Estimate - Point {group_name}')
    plt.legend()

    plt.savefig(f"./graph/estimate vs observed/graphs/graph_group_{group_name}")
    plt.show()
