import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

data = pd.read_csv(os.path.relpath("./graph/fitted vs observed/eeek_output.csv"))

data['date'] = pd.to_datetime(data['date'], unit='ms')

filtered_data = data[data['z'] != 0]

dates = filtered_data['date']
fitted = filtered_data['fitted']
observed = filtered_data['z']

plt.figure(figsize=(10, 6))

plt.plot(dates, fitted, label='Fitted', color='red', linestyle='-')

plt.scatter(dates, observed, label='observed', color='blue', s=10)

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title('Fitted vs Observed')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# plt.xticks(rotation=45)    
plt.savefig("fitted")
plt.show()