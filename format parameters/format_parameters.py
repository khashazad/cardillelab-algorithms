import json
import csv

json_files = ['v1.json', 'v2.json', 'v3.json', 'v4.json']

def get_run_title(index): 
    if index == 0:
        return 'Start from default parameters'
    elif index == 1:
        return 'Start from default parameters increased by factor of 10'
    elif index == 2:
        return 'Start from default parameters increased by factor of 100'
    elif index == 3:
        return 'Start far from default parameters'
    
with open('./format parameters/formatted_params.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['Run', 'Q1', 'Q2', 'Q3', 'R'])
    writer.writerow(['Default Parameters', 0.00125, 0.000125, 0.000125, 0.003])

    for idx, file in enumerate(json_files):
        with open(f"./format parameters/params/{file}", 'r') as jsonfile:
            data = json.load(jsonfile)

            q_diagonal = [data['process noise (Q)'][i][i] for i in range(len(data['process noise (Q)']))]

            q1 = data['process noise (Q)'][0][0]
            q2 = data['process noise (Q)'][1][1]
            q3 = data['process noise (Q)'][2][2]

            error_term = data['measurement noise (R)'][0][0]

            writer.writerow([get_run_title(idx), q1, q2, q3, error_term])