import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

data = pd.read_csv(os.path.relpath("./graph/pest_output.csv"))

data['date'] = pd.to_datetime(data['date'], unit='ms')

filtered_data = data[data['z'] != 0]

dates = filtered_data['date']
estimate = filtered_data['estimate']
z = filtered_data['z']

plt.figure(figsize=(10, 6))

plt.plot(dates, estimate, label='Estimate', color='blue', linestyle='-')

plt.scatter(dates, z, label='Z', color='red', s=10)

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title('Estimate and Z Values Over Time (Excluding Zero Values)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# plt.xticks(rotation=45)    
plt.savefig("estimate_vs_observed_graph")
plt.show()