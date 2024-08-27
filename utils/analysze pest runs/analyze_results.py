import os
import json
import shutil
import csv
from graphs import create_graphs
from rmse import calculate_rmse

all_pest_results_directory = os.path.dirname(os.path.abspath(__file__)) + "/../../pest runs/"
pest_runs = "16 points"

results_directory = os.path.join(all_pest_results_directory, pest_runs)

analysis_directory = os.path.join(results_directory, "analysis")
if os.path.exists(analysis_directory):
    shutil.rmtree(analysis_directory)

os.makedirs(analysis_directory)

def parse_initial_parameters(pest_file_path):
    initial_parameters = {}
    with open(pest_file_path, "r") as file:
        lines = file.readlines()

    for line in lines[23:27]:
        parts = line.split()
        param_name = parts[0]
        param_value = parts[3]
        initial_parameters[param_name] = param_value

    return initial_parameters

def parse_initial_and_final_objective_function(pest_file_path):
    with open(pest_file_path, "r") as file:
        lines = file.readlines()
        objective_functions = [line.split("=")[1].strip() for line in lines if "Sum of squared weighted residuals" in line]

        return float(objective_functions[0]), float(objective_functions[-1])

def parse_optimized_parameters(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

        optimized_parameters = {}

        optimized_parameters["q1"] = data["process noise (Q)"][0][0]
        optimized_parameters["q2"] = data["process noise (Q)"][1][1]
        optimized_parameters["q3"] = data["process noise (Q)"][2][2]
        optimized_parameters["r"] = data["measurement noise (R)"][0][0]

    return optimized_parameters

observation_file_path = None 

with open(os.path.join(analysis_directory, "analysis.csv"), "w", newline='') as file:
    analysis_writer = csv.writer(file)

    headers = ["title", "q1_initial", "q2_initial", "q3_initial", "r_initial", "q1_optimized", "q2_optimized", "q3_optimized", "r_optimized", "initial_objective_function", "final_objective_function"]
    analysis_writer.writerow(headers)

    data_files = {}

    for grouped_run_folder in os.listdir(results_directory):
        if grouped_run_folder == "analysis": 
            continue

        grouped_run_folder_path = os.path.join(results_directory, grouped_run_folder)
        for run in os.listdir(grouped_run_folder_path):
            run_path = os.path.join(grouped_run_folder_path, run)

            run_title = f"{grouped_run_folder} - {run}"

            data_files[run_title] = os.path.join(run_path, "pest_output.csv")

            if observation_file_path is None:
                observation_file_path = os.path.join(run_path, "observations.csv")

            initial_parameters = parse_initial_parameters(os.path.join(run_path, "eeek.pst")).values()
            optimized_parameters = parse_optimized_parameters(os.path.join(run_path, "eeek_params.json")).values()

            initial_objective_function, final_objective_function = parse_initial_and_final_objective_function(os.path.join(run_path, "eeek.rec"))

            output_row = [run_title]
            output_row.extend(optimized_parameters)
            output_row.extend(initial_parameters)
            output_row.extend([initial_objective_function, final_objective_function])

            analysis_writer.writerow(output_row)

    flags = {
        "estimate": True,
        "final_2022_fit": False,
        "final_2023_fit": False,
        "intercept_cos_sin": True,
        "residuals": True,
        "amplitude": True
    }

    calculate_rmse(data_files, observation_file_path, os.path.join(analysis_directory, "rmse"))
    create_graphs(data_files, observation_file_path, os.path.join(analysis_directory, "graphs"), flags)







