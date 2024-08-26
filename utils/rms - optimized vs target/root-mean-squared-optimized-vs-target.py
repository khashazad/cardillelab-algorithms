import pandas as pd
import numpy as np
import os
import math
import csv

script_directory = os.path.dirname(os.path.abspath(__file__))

observations_df = pd.read_csv(f'{script_directory}/data/observations.csv')
pest_output_df = pd.read_csv(f'{script_directory}/data/pest_output.csv')

merged_df = pd.merge(observations_df, pest_output_df)

grouped_df = merged_df.groupby('point')

with open(f'{script_directory}/rms.csv', 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['point', 'intercept_rms', 'cos_rms', 'sin_rms', 'sum'])

    for group_name, group_data in grouped_df:
        intercept_rms = math.sqrt(((group_data['INTP'] - group_data['intercept']) ** 2).mean())
        cos_rms = math.sqrt(((group_data['COS0'] - group_data['cos']) ** 2).mean())
        sin_rms = math.sqrt(((group_data['SIN0'] - group_data['sin']) ** 2).mean())
        sum = intercept_rms + cos_rms + sin_rms

        csv_writer.writerow([group_name, intercept_rms, cos_rms, sin_rms, sum])  

