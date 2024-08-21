import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

data = pd.read_csv(os.path.relpath("./graph/final intercept vs Target/eeek_output.csv"))

data['date'] = pd.to_datetime(data['date'], unit='ms')

filtered_data = data[data['z'] != 0]

dates = filtered_data['date']
intercept = filtered_data['INTP']
target = filtered_data['target']


plt.figure(figsize=(10, 6))

plt.plot(dates, target, label='Target', color='blue', linestyle='-')

plt.plot(dates, intercept, label='Intercept', color='green', linestyle='-')

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title('Target Intercept vs final intercept')
# plt.xlabel('Date')
# plt.ylabel('Values')
plt.legend()

# plt.xticks(rotation=45)    
plt.savefig("./graph/estimate vs observed/graph")
plt.show()