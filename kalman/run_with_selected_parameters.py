import os
import shutil
import ee.geometry, ee
import pandas as pd
import math
from lib.image_collections import COLLECTIONS
from utils.visualization.charts import generate_charts_comparing_runs
from utils.filesystem import delete_existing_directory_and_create_new, read_file
from utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from utils import utils
from datetime import datetime
from pest_eeek import main as run_eeek
import csv
import concurrent.futures

VERSION = 2

POINT_SET = 1
POINTS_COUNT = 15

graph_flags = {
    "estimate": True,
    "final_2022_fit": False,
    "final_2023_fit": False,
    "intercept_cos_sin": True,
    "residuals": True,
    "amplitude": True,
}

script_directory = os.path.dirname(os.path.realpath(__file__))
parent_run_directory = f"{script_directory}/eeek runs/set {POINT_SET} - {POINTS_COUNT} points/version {VERSION}/"

point_set_directory_path = f"{script_directory}/points/sets/{POINT_SET}"

os.makedirs(parent_run_directory, exist_ok=True)


def run_eeek_with_selected_parameters(parameter_file_path):
    global parent_run_directory

    title = os.path.basename(parameter_file_path).replace("_", " ").split(".")[0]
    local_run_directory = os.path.join(parent_run_directory, "runs", title)

    os.makedirs(local_run_directory, exist_ok=True)

    os.chdir(local_run_directory)

    input_file_path = os.path.join(local_run_directory, "eeek_input.csv")
    output_file_path = os.path.join(local_run_directory, "eeek_output.csv")
    points_file_path = os.path.join(local_run_directory, "points.csv")

    shutil.copy(parameter_file_path, input_file_path)
    shutil.copy(os.path.join(parent_run_directory, "points.csv"), points_file_path)

    args = {
        "input": input_file_path,
        "output": output_file_path,
        "points": points_file_path,
        "num_sinusoid_pairs": 1,
        "collection": "L8_L9_2022_2023",
        "include_intercept": True,
        "store_measurement": True,
        "store_estimate": True,
        "store_date": True,
        "include_slope": False,
        "store_amplitude": False,
    }

    run_eeek(args)


def run_eeek_with_default_parameters():
    global parent_run_directory
    global script_directory

    default_params_directory = os.path.join(
        script_directory, "eeek runs", "parameter set", "default"
    )

    if os.path.exists(os.path.join(parent_run_directory, "default runs")):
        return

    def process_parameter_file(parameter_file_path):
        title = os.path.basename(parameter_file_path).replace("_", " ").split(".")[0]
        local_run_directory = os.path.join(parent_run_directory, "default runs", title)

        os.makedirs(local_run_directory, exist_ok=True)

        os.chdir(local_run_directory)

        input_file_path = os.path.join(local_run_directory, "eeek_input.csv")
        output_file_path = os.path.join(local_run_directory, "eeek_output.csv")
        points_file_path = os.path.join(local_run_directory, "points.csv")

        shutil.copy(parameter_file_path, input_file_path)
        shutil.copy(os.path.join(parent_run_directory, "points.csv"), points_file_path)

        args = {
            "input": input_file_path,
            "output": output_file_path,
            "points": points_file_path,
            "num_sinusoid_pairs": 1,
            "collection": "L8_L9_2022_2023",
            "include_intercept": True,
            "store_measurement": True,
            "store_estimate": True,
            "store_date": True,
            "include_slope": False,
            "store_amplitude": False,
        }

        run_eeek(args)

    # for parameter_file in os.listdir(default_params_directory):
    #     process_parameter_file(os.path.join(default_params_directory, parameter_file))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            process_parameter_file,
            [
                os.path.join(default_params_directory, parameter_file)
                for parameter_file in os.listdir(default_params_directory)
            ],
        )


def parse_point_coordinates():
    global point_set_directory_path

    point_coordinates = []

    for folder in os.listdir(point_set_directory_path):
        folder_path = os.path.join(point_set_directory_path, folder)
        if os.path.isdir(folder_path):
            point_coordinates.extend(
                [
                    (float(file.split(",")[0][1:]), float(file.split(",")[1][:-5]))
                    for file in os.listdir(folder_path)
                ]
            )

    return sorted(point_coordinates, key=lambda x: (x[0], x[1]))


def get_dates_from_image_collection(year, coords):
    timestamps = [
        image["properties"]["millis"]
        for image in COLLECTIONS[f"L8_L9_2022_2023"]
        .filterBounds(ee.Geometry.Point(coords))
        .getInfo()["features"]
    ]

    return [
        timestamp
        for timestamp in timestamps
        if datetime.fromtimestamp(timestamp / 1000.0).year == year
    ]


def harmonic_trend_coefficients(collection, coords):
    modality = {
        "constant": True,
        "linear": False,
        "unimodal": True,
        "bimodal": False,
        "trimodal": False,
    }

    image_collection = ee.ImageCollection(
        collection.filterBounds(ee.Geometry.Point(coords))
    )

    reduced_image_collection_with_harmonics = (
        add_harmonic_bands_via_modality_dictionary(image_collection, modality)
    )

    harmonic_independent_variables = (
        determine_harmonic_independents_via_modality_dictionary(modality)
    )

    harmonic_one_time_regression = fit_harmonic_to_collection(
        reduced_image_collection_with_harmonics, "swir", harmonic_independent_variables
    )
    fitted_coefficients = harmonic_one_time_regression["harmonic_trend_coefficients"]

    return fitted_coefficients


def get_fitted_coefficients_for_point(collection, coords, year):
    request = utils.build_request(coords)
    request["expression"] = harmonic_trend_coefficients(collection, coords)
    coefficients = utils.compute_pixels_wrapper(request)

    image_dates = get_dates_from_image_collection(year, coords)

    return {
        "intercept": coefficients[0],
        "cos": coefficients[1],
        "sin": coefficients[2],
        "dates": image_dates,
    }


def fitted_coefficients_and_dates(points, fitted_coefficiets_filename):

    output_list = []
    coefficients_by_point = {}

    with open(fitted_coefficiets_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [
                "point",
                "longitude",
                "latitude",
                "intercept_2022",
                "cos_2022",
                "sin_2022",
                "intercept_2023",
                "cos_2023",
                "sin_2023",
            ]
        )
        for i, point in enumerate(points):
            coefficients_by_point[i] = {
                "coordinates": (point[0], point[1]),
                "2022": get_fitted_coefficients_for_point(
                    COLLECTIONS["L8_L9_2022"].filterBounds(
                        ee.Geometry.Point(point[0], point[1])
                    ),
                    (point[0], point[1]),
                    2022,
                ),
                "2023": get_fitted_coefficients_for_point(
                    COLLECTIONS["L8_L9_2023"].filterBounds(
                        ee.Geometry.Point(point[0], point[1])
                    ),
                    (point[0], point[1]),
                    2023,
                ),
            }

            csv_writer.writerow(
                [
                    i,
                    point[0],
                    point[1],
                    coefficients_by_point[i]["2022"]["intercept"],
                    coefficients_by_point[i]["2022"]["cos"],
                    coefficients_by_point[i]["2022"]["sin"],
                    coefficients_by_point[i]["2023"]["intercept"],
                    coefficients_by_point[i]["2023"]["cos"],
                    coefficients_by_point[i]["2023"]["sin"],
                ]
            )

            output_list.append(coefficients_by_point[i])

    return output_list


def create_points_file(points_filename, coefficients_by_point):
    with open(points_filename, "w", newline="") as file:
        for idx, point in enumerate(coefficients_by_point):
            longitude = point["coordinates"][0]
            latitude = point["coordinates"][1]

            intercept = point["2022"]["intercept"]
            cos = point["2022"]["cos"]
            sin = point["2022"]["sin"]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")


def build_observations(coefficients_by_point, output_filename):
    observations = []

    with open(output_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["point", "date", "intercept", "cos", "sin", "estimate"])
        for index, dic in enumerate(coefficients_by_point):
            observation_index = 1

            def create_observation_from_coefficients(dates, intercept, cos, sin):
                nonlocal observation_index
                for date in dates:
                    time = (
                        pd.Timestamp(date, unit="ms") - pd.Timestamp("2016-01-01")
                    ).total_seconds() / (365.25 * 24 * 60 * 60)
                    phi = 6.283 * time
                    phi_cos = math.cos(phi)
                    phi_sin = math.sin(phi)
                    estimate = intercept + cos * phi_cos + sin * phi_sin

                    observations.append(
                        (f"intercept_{int(index)}_{observation_index}", intercept)
                    )
                    observations.append((f"cos_{int(index)}_{observation_index}", cos))
                    observations.append((f"sin_{int(index)}_{observation_index}", sin))
                    observations.append(
                        (f"estimate_{int(index)}_{observation_index}", estimate)
                    )
                    csv_writer.writerow([index, date, intercept, cos, sin, estimate])
                    observation_index += 1

            coefficients_2022 = dic["2022"]
            coefficients_2023 = dic["2023"]

            create_observation_from_coefficients(
                coefficients_2022["dates"],
                coefficients_2022["intercept"],
                coefficients_2022["cos"],
                coefficients_2022["sin"],
            )
            create_observation_from_coefficients(
                coefficients_2023["dates"],
                coefficients_2023["intercept"],
                coefficients_2023["cos"],
                coefficients_2023["sin"],
            )

        return observations


def analyze_results(run_directory):
    global parent_run_directory

    analysis_directory = os.path.join(
        parent_run_directory, "analysis", os.path.basename(run_directory)
    )
    delete_existing_directory_and_create_new(analysis_directory)

    observations_file_path = os.path.join(parent_run_directory, "observations.csv")

    data_files = {
        os.path.basename(run_directory).split(".")[0]: os.path.join(
            run_directory, "eeek_output.csv"
        )
    }

    default_runs_directories = [
        run
        for run in os.listdir(os.path.join(parent_run_directory, "default runs"))
        if os.path.isdir(os.path.join(parent_run_directory, "default runs", run))
    ]

    for run in default_runs_directories:
        run_directory = os.path.join(parent_run_directory, "default runs", run)

        data_files[run.split(".")[0]] = os.path.join(run_directory, "eeek_output.csv")

    generate_charts_comparing_runs(
        data_files, observations_file_path, analysis_directory, graph_flags
    )

    parameters = []
    for param_set in [os.path.dirname(param) for param in data_files.values()]:
        Q, R, P = read_file(os.path.join(param_set, "eeek_input.csv"))
        Q = [float(q) for q in Q.split(",")]
        parameters.append([param_set.split("/")[-1], Q[0], Q[4], Q[8], float(R)])

    with open(os.path.join(analysis_directory, "parameters.csv"), "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Set", "Q1", "Q5", "Q9", "R"])
        csv_writer.writerows(parameters)


if __name__ == "__main__":
    fitted_coefficiets_filename = parent_run_directory + "fitted_coefficients.csv"
    points_filename = parent_run_directory + "points.csv"
    observations_filename = parent_run_directory + "observations.csv"

    points = parse_point_coordinates()

    if not os.path.exists(fitted_coefficiets_filename):
        fitted_coefficiets_by_point = fitted_coefficients_and_dates(
            points, fitted_coefficiets_filename
        )

    if not os.path.exists(points_filename):
        create_points_file(points_filename, fitted_coefficiets_by_point)

    if not os.path.exists(observations_filename):
        observations = build_observations(
            fitted_coefficiets_by_point, observations_filename
        )

    parameter_sets_directory = os.path.join(
        script_directory, "eeek runs", "parameter set", "selected"
    )
    for parameter_set in os.listdir(parameter_sets_directory):
        parameter_set_path = os.path.join(parameter_sets_directory, parameter_set)
        run_eeek_with_selected_parameters(parameter_set_path)

    run_eeek_with_default_parameters()

    for result_directory in os.listdir(os.path.join(parent_run_directory, "runs")):
        result_directory_path = os.path.join(
            parent_run_directory, "runs", result_directory
        )
        analyze_results(result_directory_path)
