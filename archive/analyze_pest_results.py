import os
import json
import shutil
import csv
import argparse
from utils.charts import create_graphs
from utils.root_mean_squared_e import calculate_rmse
from pest_eeek import main as run_eeek

script_directory = os.path.dirname(os.path.abspath(__file__))
all_pest_results_directory = os.path.join(script_directory, "pest runs/")


parser = argparse.ArgumentParser()
parser.add_argument("--point_set", default="set 2 - 20 points", help="Specify the pest run directory.")
args = parser.parse_args()

point_set = args.point_set

results_directory = os.path.join(all_pest_results_directory, point_set)

def delete_existing_directory_and_create_new(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


analysis_directory = os.path.join(results_directory, "analysis")

delete_existing_directory_and_create_new(analysis_directory)

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

def create_graphs_with_custom_run_output(run_path):
    analysis_directory = os.path.join(run_path, "analysis/")
    delete_existing_directory_and_create_new(analysis_directory)

    pest_output_file_path = os.path.join(run_path, "pest_output.csv")
    observations_file_path = os.path.join(run_path, "observations.csv")
    default_runs_directory = os.path.join(run_path, "default runs/")
    
    custom_run_data_files = {
        "PEST Optimized Output": pest_output_file_path
    }

    for eeek_run_folder in os.listdir(default_runs_directory):
        if eeek_run_folder == "analysis" or eeek_run_folder == ".DS_Store":
            continue

        eeek_run_folder_path = os.path.join(default_runs_directory, eeek_run_folder)

        custom_run_data_files[eeek_run_folder.split(".")[0]] =  os.path.join(eeek_run_folder_path, "eeek_output.csv")

    flags = {
        "estimate": True,
        "final_2022_fit": False,
        "final_2023_fit": False,
        "intercept_cos_sin": True,
        "residuals": True,
        "amplitude": True
    }

    create_graphs(custom_run_data_files, observations_file_path, analysis_directory, flags)

observation_file_path = None 

with open(os.path.join(analysis_directory, "analysis.csv"), "w", newline='') as file:
    analysis_writer = csv.writer(file)

    headers = ["title", "q1_initial", "q2_initial", "q3_initial", "r_initial", "q1_optimized", "q2_optimized", "q3_optimized", "r_optimized", "initial_objective_function", "final_objective_function"]
    analysis_writer.writerow(headers)

    data_files = {}

    for run in os.listdir(results_directory):
        if run == ".DS_Store" or run == "analysis":
            continue

        run_path = os.path.join(results_directory, run)

        create_graphs_with_custom_run_output(run_path)

        run_title = f"{point_set.split(" - ")[1]} - {run}"

        data_files[run_title] = os.path.join(run_path, "pest_output.csv")

        if observation_file_path is None:
            observation_file_path = os.path.join(run_path, "observations.csv")

        initial_parameters = parse_initial_parameters(os.path.join(run_path, "eeek.pst")).values()
        optimized_parameters = parse_optimized_parameters(os.path.join(run_path, "eeek_params.json")).values()

        initial_objective_function, final_objective_function = parse_initial_and_final_objective_function(os.path.join(run_path, "eeek.rec"))

        output_row = [run_title]
        output_row.extend(initial_parameters)
        output_row.extend(optimized_parameters)
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







