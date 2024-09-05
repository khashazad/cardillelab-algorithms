import os
from utils.charts import generate_charts_comparing_runs
from utils.filesystem import delete_existing_directory_and_create_new


def analyze_results(run_directory, graph_flags):

    analysis_directory = os.path.join(run_directory, "analysis")
    delete_existing_directory_and_create_new(analysis_directory)

    observations_file_path = os.path.join(run_directory, "observations.csv")

    data_files = {
        os.path.basename(run_directory).split(".")[0]: os.path.join(
            run_directory, "eeek_output.csv"
        )
    }

    generate_charts_comparing_runs(
        data_files, observations_file_path, analysis_directory, graph_flags
    )