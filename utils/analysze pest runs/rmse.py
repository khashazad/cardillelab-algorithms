import pandas as pd
import numpy as np
import os
import math
import csv
import shutil

def calculate_rmse(data_files, observations_path, output_folder):
    observations_df = pd.read_csv(observations_path)

    rmse = lambda x, y: math.sqrt(((x - y) ** 2).mean())

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)


    rmse_by_run = []

    for title, file_path in data_files.items():
        pest_output_df = pd.read_csv(file_path)
        merged_df = pd.merge(observations_df, pest_output_df, on='point')

        grouped_df = merged_df.groupby('point')

        run_directory_path = os.path.join(output_folder, title)
        os.makedirs(run_directory_path)

        output_file_path = os.path.join(run_directory_path, f'intercept_cos_sin_rmse.csv')
        with open(output_file_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['point', 'intercept_rmse', 'cos_rmse', 'sin_rmse', 'sum_rmse'])

            records = []

            for group_name, group_data in grouped_df:
                intercept_rms = rmse(group_data['INTP'], group_data['intercept'])
                cos_rms = rmse(group_data['COS0'], group_data['cos'])
                sin_rms = rmse(group_data['SIN0'], group_data['sin'])
                sum_rms = intercept_rms + cos_rms + sin_rms
                
                records.append([group_name, intercept_rms, cos_rms, sin_rms, sum_rms])

                csv_writer.writerow([group_name, intercept_rms, cos_rms, sin_rms, sum_rms])  

            total_intercept_rms =  rmse(merged_df['INTP'], merged_df['intercept'])
            total_cos_rms =  rmse(merged_df['COS0'], merged_df['cos'])
            total_sin_rms =  rmse(merged_df['SIN0'], merged_df['sin'])
            total_sum_rms = total_intercept_rms + total_cos_rms + total_sin_rms
            
            rmse_by_run.append([title, total_intercept_rms, total_cos_rms, total_sin_rms, total_sum_rms])
            csv_writer.writerow(['total', total_intercept_rms, total_cos_rms, total_sin_rms, total_sum_rms])

            df = pd.DataFrame(records, columns=['point', 'intercept_rmse', 'cos_rmse', 'sin_rmse', 'sum_rmse'])
            sorted_df = df.sort_values(by='sum_rmse', ascending=False)
            sorted_df.to_csv(os.path.join(run_directory_path, 'sorted_rmse.csv'), index=False)

    rmse_by_run_df = pd.DataFrame(rmse_by_run, columns=['run', 'intercept_rmse', 'cos_rmse', 'sin_rmse', 'sum_rmse'])   
    sorted_rmse_by_run_df = rmse_by_run_df.sort_values(by='sum_rmse', ascending=False)
    sorted_rmse_by_run_df.to_csv(os.path.join(output_folder, 'rmse_by_run.csv'), index=False)
            