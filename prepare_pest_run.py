import ee.geometry, ee
import pandas as pd
from lib.image_collections import COLLECTIONS
from utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    fit_harmonic_to_collection,
    determine_harmonic_independents_via_modality_dictionary,
)
from utils import utils
from pprint import pprint
from datetime import datetime
from pest_eeek import main as run_eeek
import csv
import os
import math
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.filesystem import delete_existing_directory_and_create_new, read_json
from utils.pest.pest_file_builder import (
    create_control_file,
    append_observations_to_control_file,
    create_instructions_file,
    append_model_and_io_sections_to_control_file,
    create_template_file,
    create_model_file,
)
import argparse

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

script_directory = os.path.dirname(os.path.realpath(__file__))

POINT_SET = 5
POINTS_COUNT = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "--initial_params", default="v1", help="Version of the initial parameters."
)
args = parser.parse_args()
INITIAL_PARAMS_VERSION = args.initial_params

OBSERVATIONS_FLAGS = {
    "intercept": True,
    "cos": False,
    "sin": False,
    "estimate": False,
    "amplitude": True,
}

parameters = f"{script_directory}/pest configuration/default - initial {INITIAL_PARAMS_VERSION}.json"
pest_run_directory = f"{script_directory}/pest runs/set {POINT_SET} - {POINTS_COUNT} points/IA - initial params {INITIAL_PARAMS_VERSION}/"

point_set_directory_path = f"{script_directory}/points/sets/{POINT_SET}"

if os.path.exists(pest_run_directory):
    print("Output directory already exists. Exiting to prevent overwriting.")
    exit()
os.makedirs(pest_run_directory)


def run_eeek_with_default_parameters():
    global pest_run_directory
    global script_directory
    default_runs_directory = os.path.join(pest_run_directory, "default runs/")

    delete_existing_directory_and_create_new(default_runs_directory)

    default_params_directory = os.path.join(script_directory, "eeek params")

    for run_parameters_file in os.listdir(default_params_directory):
        run_parameters_file_path = os.path.join(
            default_params_directory, run_parameters_file
        )
        title = run_parameters_file.replace("_", " ").split(".")[0]

        run_directory = os.path.join(default_runs_directory, title)

        os.mkdir(run_directory)
        os.chdir(run_directory)

        input_file_path = os.path.join(run_directory, "eeek_input.csv")
        output_file_path = os.path.join(run_directory, "eeek_output.csv")
        points_file_path = os.path.join(run_directory, "points.csv")

        shutil.copy(run_parameters_file_path, input_file_path)
        shutil.copy(os.path.join(pest_run_directory, "points.csv"), points_file_path)

        args = {
            "input": input_file_path,
            "output": output_file_path,
            "points": points_file_path,
            "num_sinusoid_pairs": 1,
            "collection": "L8_L9_2022_2023",
            "include_intercept": True,
            "store_measurement": True,
            "store_estimate": True,
            "store_amplitude": True,
            "store_date": True,
            "include_slope": False,
        }

        run_eeek(args)


def build_observations(coefficients_by_point, output_filename):
    observations = []

    with open(output_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["point", "date", "intercept", "cos", "sin", "estimate", "amplitude"]
        )
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
                    amplitude = math.sqrt(cos**2 + sin**2)

                    observations.append(
                        (f"intercept_{int(index)}_{observation_index}", intercept)
                    )
                    observations.append((f"cos_{int(index)}_{observation_index}", cos))
                    observations.append((f"sin_{int(index)}_{observation_index}", sin))
                    observations.append(
                        (f"estimate_{int(index)}_{observation_index}", estimate)
                    )
                    observations.append(
                        (f"amplitude_{int(index)}_{observation_index}", amplitude)
                    )
                    csv_writer.writerow(
                        [index, date, intercept, cos, sin, estimate, amplitude]
                    )
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


def create_points_file(points_filename, coefficients_by_point):
    with open(points_filename, "w", newline="") as file:
        for idx, point in enumerate(coefficients_by_point):
            longitude = point["coordinates"][0]
            latitude = point["coordinates"][1]

            intercept = point["2022"]["intercept"]
            cos = point["2022"]["cos"]
            sin = point["2022"]["sin"]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")


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


def generate_measurements_and_target_fit_graphs(
    observations_filename, measurements_filename
):
    output_directory = os.path.join(pest_run_directory, "measurements and target fit")

    delete_existing_directory_and_create_new(output_directory)

    observations = pd.read_csv(observations_filename)
    measurements = pd.read_csv(measurements_filename)

    observations["measurement"] = measurements["z"]

    observations["date"] = pd.to_datetime(observations["date"], unit="ms")

    grouped_observations = observations.groupby("point")

    for point, data in grouped_observations:
        fig, axs = plt.subplots(figsize=(12, 8))

        data = data[data["measurement"] != 0]

        axs.scatter(
            data["date"], data["measurement"], label="Measurement", color="red", s=10
        )
        axs.plot(
            data["date"],
            data["estimate"],
            label="Target Fit",
            color="green",
            linestyle="-",
        )

        axs.xaxis.set_major_locator(mdates.AutoDateLocator())
        axs.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plot_filename = os.path.join(output_directory, f"point_{point}.png")
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(plot_filename)
        plt.close(fig)


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


if __name__ == "__main__":
    control_filename = pest_run_directory + "eeek.pst"
    instructions_filename = pest_run_directory + "output.ins"
    template_filename = pest_run_directory + "input.tpl"
    points_filename = pest_run_directory + "points.csv"
    model_filename = pest_run_directory + "model.bat"
    fitted_coefficiets_filename = pest_run_directory + "fitted_coefficients.csv"
    observations_filename = pest_run_directory + "observations.csv"

    points = parse_point_coordinates()

    fitted_coefficiets_by_point = fitted_coefficients_and_dates(
        points, fitted_coefficiets_filename
    )

    observations = build_observations(
        fitted_coefficiets_by_point, observations_filename
    )

    create_control_file(
        read_json(parameters),
        control_filename,
        int(
            len(observations)
            * (
                len([flag for flag in OBSERVATIONS_FLAGS.values() if flag])
                / len(OBSERVATIONS_FLAGS.values())
            )
        ),
    )
    append_observations_to_control_file(
        observations, control_filename, OBSERVATIONS_FLAGS
    )
    create_instructions_file(observations, instructions_filename, OBSERVATIONS_FLAGS)
    append_model_and_io_sections_to_control_file(control_filename)
    create_template_file(template_filename)
    create_model_file(model_filename)

    create_points_file(points_filename, fitted_coefficiets_by_point)
    run_eeek_with_default_parameters()
    generate_measurements_and_target_fit_graphs(
        observations_filename,
        os.path.join(
            pest_run_directory, "default runs/default javascript (v1)/eeek_output.csv"
        ),
    )

    print(f"Pest files has been created.")
