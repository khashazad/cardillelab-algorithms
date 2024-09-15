import os
import json
import shutil
import csv
import argparse
from utils.charts import create_graphs, generate_charts_comparing_runs
from utils.root_mean_squared_e import calculate_rmse
from pest_eeek import main as run_eeek
from utils.filesystem import delete_existing_directory_and_create_new
from utils.pest_output_analysis import parse_initial_parameters, parse_optimized_parameters, parse_initial_and_final_objective_function

script_directory = os.path.dirname(os.path.abspath(__file__))
all_pest_results_directory = os.path.join(script_directory, "pest runs")

parser = argparse.ArgumentParser()
parser.add_argument("--point_set", default="set 2 - 20 points", help="Specify the pest run directory.")
args = parser.parse_args()

point_set = args.point_set

results_directory = os.path.join(all_pest_results_directory, point_set)
analysis_directory = os.path.join(all_pest_results_directory, "analysis", point_set)
os.makedirs(analysis_directory, exist_ok=True)
observation_file_path = None 

graph_flags = {
    "estimate": True,
    "final_2022_fit": False,
    "final_2023_fit": False,
    "intercept_cos_sin": True,
    "residuals": True,
    "amplitude": True
}

def compare_run_to_default_runs(run_path):
    run_title = f"{run_path.split("/")[1]}"
    analysis_directory = os.path.join(script_directory, "pest runs", "analysis", point_set, "runs", run_title)
    delete_existing_directory_and_create_new(analysis_directory)

    pest_output_file_path = os.path.join(run_path, "pest_output.csv")
    observations_file_path = os.path.join(run_path, "observations.csv")
    default_runs_directory = os.path.join(run_path, "default runs")
    
    data_files = {
        "optimized": pest_output_file_path
    }

    for default_run in os.listdir(default_runs_directory):
        if default_run == "analysis" or default_run == ".DS_Store":
            continue

        default_run_directory = os.path.join(default_runs_directory, default_run)

        data_files[default_run.split(".")[0]] =  os.path.join(default_run_directory, "eeek_output.csv")

    generate_charts_comparing_runs(data_files, observations_file_path, analysis_directory, graph_flags)
with open(os.path.join(analysis_directory, "analysis.csv"), "w", newline='') as file:
    analysis_writer = csv.writer(file)

    headers = ["title", "q1_initial", "q1_optimized", "q5_initial", "q5_optimized", "q9_initial", "q9_optimized", "r_initial", "r_optimized", "initial_objective_function", "final_objective_function"]
    analysis_writer.writerow(headers)

    data_files = {}

    for run in os.listdir(results_directory):
        if run == ".DS_Store" or run == "analysis":
            continue

        run_path = os.path.join(results_directory, run)

        compare_run_to_default_runs(run_path)

        run_title = f"{point_set.split(" - ")[1]} - {run}"

        data_files[run_title] = os.path.join(run_path, "pest_output.csv")

        if observation_file_path is None:
            observation_file_path = os.path.join(run_path, "observations.csv")

        initial_q1, initial_q5, initial_q9, initial_r = parse_initial_parameters(os.path.join(run_path, "eeek.pst"))
        optimized_q1, optimized_q5, optimized_q9, optimized_r = parse_optimized_parameters(os.path.join(run_path, "eeek_params.json"))
        initial_objective_function, final_objective_function = parse_initial_and_final_objective_function(os.path.join(run_path, "eeek.rec"))

        analysis_writer.writerow([run_title, initial_q1, optimized_q1, initial_q5, optimized_q5, initial_q9, optimized_q9, initial_r, optimized_r, initial_objective_function, final_objective_function])

    # calculate_rmse(data_files, observation_file_path, os.path.join(analysis_directory, "rmse"))
    # create_graphs(data_files, observation_file_path, os.path.join(analysis_directory, "graphs"), graph_flags)







