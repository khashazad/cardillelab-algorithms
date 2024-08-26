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

with open(f'{script_directory}/rms.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['point', 'intercept_rms', 'cos_rms', 'sin_rms', 'sum'])

    rmse = lambda x, y: math.sqrt(((x - y) ** 2).mean())

    records = []

    for group_name, group_data in grouped_df:
        intercept_rms = rmse(group_data['INTP'], group_data['intercept'])
        cos_rms = rmse(group_data['COS0'], group_data['cos'])
        sin_rms = rmse(group_data['SIN0'], group_data['sin'])
        sum = intercept_rms + cos_rms + sin_rms
        
        records.append([group_name, intercept_rms, cos_rms, sin_rms, sum])

   

        csv_writer.writerow([group_name, intercept_rms, cos_rms, sin_rms, sum])  

    df = pd.DataFrame(records, columns=['point', 'intercept_rmse', 'cos_rmse', 'sin_rmse', 'sum_rmse'])

    
    top_5_rows = df.nlargest(15, 'sum_rmse')[['point', 'sum_rmse', 'intercept_rmse', 'cos_rmse', 'sin_rmse']]
    print(top_5_rows)

