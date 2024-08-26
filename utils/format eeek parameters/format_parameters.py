import json
import csv
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

json_files = [file for file in os.listdir(f'{script_directory}/params') if file.endswith('.json')]
    
with open(f'{script_directory}/formatted_params.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['title', 'Q1', 'Q2', 'Q3', 'R'])
    # writer.writerow(['Default Parameters', 0.00125, 0.000125, 0.000125, 0.003])

    for idx, file in enumerate(json_files):
        with open(f"{script_directory}/params/{file}", 'r') as jsonfile:
            data = json.load(jsonfile)

            q_diagonal = [data['process noise (Q)'][i][i] for i in range(len(data['process noise (Q)']))]

            q1 = round(float(data['process noise (Q)'][0][0]), 5)
            q2 = round(float(data['process noise (Q)'][1][1]), 5)
            q3 = round(float(data['process noise (Q)'][2][2]), 5)

            error_term = round(float(data['measurement noise (R)'][0][0]), 5)

            writer.writerow([file.split(".")[0], q1, q2, q3, error_term])