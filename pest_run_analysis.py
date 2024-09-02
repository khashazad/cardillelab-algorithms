import os
import csv
import argparse
from utils.charts import generate_charts_comparing_runs
from utils.root_mean_squared_error import calculate_rmse
from utils.filesystem import delete_existing_directory_and_create_new
from utils.pest_output_analysis import (
    parse_initial_parameters,
    parse_optimized_parameters,
    parse_initial_and_final_objective_function,
)
from concurrent.futures import ProcessPoolExecutor

script_directory = os.path.dirname(os.path.abspath(__file__))
all_pest_results_directory = os.path.join(script_directory, "pest runs")

parser = argparse.ArgumentParser()
parser.add_argument("--point_set", default="4", help="Specify the pest run directory.")
args = parser.parse_args()

point_set = args.point_set

point_set_title = [
    directory
    for directory in os.listdir(all_pest_results_directory)
    if f"set {point_set}" in directory
][0]

results_directory = os.path.join(all_pest_results_directory, point_set_title)
analysis_directory = os.path.join(
    all_pest_results_directory, "analysis", point_set_title
)

delete_existing_directory_and_create_new(analysis_directory)
observation_file_path = None

graph_flags = {
    "estimate": True,
    "final_2022_fit": False,
    "final_2023_fit": False,
    "intercept_cos_sin": True,
    "residuals": True,
    "amplitude": True,
}


def compare_run_to_default_runs(run_path):
    run_title = os.path.basename(run_path)
    analysis_directory = os.path.join(
        script_directory, "pest runs", "analysis", point_set_title, "runs", run_title
    )
    delete_existing_directory_and_create_new(analysis_directory)

    pest_output_file_path = os.path.join(run_path, "pest_output.csv")
    observations_file_path = os.path.join(run_path, "observations.csv")
    default_runs_directory = os.path.join(run_path, "default runs")

    data_files = {"optimized": pest_output_file_path}

    for default_run in os.listdir(default_runs_directory):
        if default_run == "analysis" or default_run == ".DS_Store":
            continue

        default_run_directory = os.path.join(default_runs_directory, default_run)

        data_files[default_run.split(".")[0]] = os.path.join(
            default_run_directory, "eeek_output.csv"
        )

    generate_charts_comparing_runs(
        data_files, observations_file_path, analysis_directory, graph_flags
    )


def calculate_and_write_rmse(run_paths, observation_file_path, analysis_directory):
    rmse_folder = os.path.join(analysis_directory, "rmse")
    delete_existing_directory_and_create_new(rmse_folder)

    for run_path in run_paths:
        title = os.path.basename(run_path)
        rmses = calculate_rmse(
            os.path.join(run_path, "pest_output.csv"), observation_file_path
        )
        rmses.sort(key=lambda x: (x[0] != "total", -x[1], -x[2], -x[3]))

        with open(os.path.join(rmse_folder, f"{title}.csv"), "w", newline="") as file:
            rmse_writer = csv.writer(file)
            rmse_writer.writerow(
                ["point", "intercept_rmse", "cos_rmse", "sin_rmse", "sum_rmse"]
            )

            rmse_writer.writerows(rmses)


def write_initial_and_final_param_values(run_analysis_results):
    with open(
        os.path.join(analysis_directory, "analysis.csv"), "w", newline=""
    ) as file:
        analysis_writer = csv.writer(file)

        headers = [
            "title",
            "q1_initial",
            "q5_initial",
            "q9_initial",
            "r_initial",
            "q1_optimized",
            "q5_optimized",
            "q9_optimized",
            "r_optimized",
            "initial_objective_function",
            "final_objective_function",
        ]
        analysis_writer.writerow(headers)

        analysis_writer.writerows(run_analysis_results)


if __name__ == "__main__":
    run_outputs = {}

    run_paths = [
        os.path.join(results_directory, run)
        for run in os.listdir(results_directory)
        if run not in [".DS_Store", "analysis"]
    ]

    initial_vs_final_params = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(compare_run_to_default_runs, run_paths)

    for run_path in run_paths:
        run_title = os.path.basename(run_path)

        run_outputs[run_title] = os.path.join(run_path, "pest_output.csv")

        if observation_file_path is None:
            observation_file_path = os.path.join(run_path, "observations.csv")

        initial_q1, initial_q5, initial_q9, initial_r = parse_initial_parameters(
            os.path.join(run_path, "eeek.pst")
        )
        optimized_q1, optimized_q5, optimized_q9, optimized_r = (
            parse_optimized_parameters(os.path.join(run_path, "eeek_params.json"))
        )
        initial_objective_function, final_objective_function = (
            parse_initial_and_final_objective_function(
                os.path.join(run_path, "eeek.rec")
            )
        )

        initial_vs_final_params.append(
            [
                run_title,
                initial_q1,
                initial_q5,
                initial_q9,
                initial_r,
                optimized_q1,
                optimized_q5,
                optimized_q9,
                optimized_r,
                initial_objective_function,
                final_objective_function,
            ]
        )

    write_initial_and_final_param_values(initial_vs_final_params)

    calculate_and_write_rmse(run_paths, observation_file_path, analysis_directory)

    grouped_run_outputs = {}
    for title, path in run_outputs.items():
        prefix = title.split(' - ')[0]
        if prefix not in grouped_run_outputs:
            grouped_run_outputs[prefix] = {}
        grouped_run_outputs[prefix][title] = path

    for prefix, runs in grouped_run_outputs.items():
        generate_charts_comparing_runs(
            runs,
            observation_file_path,
            os.path.join(analysis_directory, "charts", prefix),
            graph_flags,
        )
