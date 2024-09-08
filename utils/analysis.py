import os
from utils.charts import generate_charts_comparing_runs
from utils.filesystem import delete_existing_directory_and_create_new


def analyze_results(data_file, observations_file_path, analysis_directory, graph_flags):
    generate_charts_comparing_runs(
        {"run": data_file}, observations_file_path, analysis_directory, graph_flags
    )