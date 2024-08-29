import json
import ee.geometry, ee
import pandas as pd
from eeek.image_collections import COLLECTIONS
from eeek.harmonic_utils import add_harmonic_bands, fit_harmonic_to_collection, determine_harmonic_independents_via_modality_dictionary
from eeek import utils
from pprint import pprint
from datetime import datetime
from pest_eeek import main as run_eeek
import csv
import os
import math
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

script_directory = os.path.dirname(os.path.realpath(__file__))

parameters = f"{script_directory}/pest configuration/default.json"

points_coordinates = f"{script_directory}/points/points-filtered.json"

pest_run_directory = f"{script_directory}/pest runs/15 points/initial params v1/"

if os.path.exists(pest_run_directory):
    print("Output directory already exists. Exiting to prevent overwriting.")

    shutil.rmtree(pest_run_directory)
    # exit()

os.makedirs(pest_run_directory)

def delete_existing_directory_and_create_new(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def run_eeek_with_default_parameters():
    global pest_run_directory
    global script_directory
    default_runs_directory = os.path.join(pest_run_directory, "default runs/")

    delete_existing_directory_and_create_new(default_runs_directory)

    default_params_directory = os.path.join(script_directory, "eeek params")

    for run_parameters_file in os.listdir(default_params_directory):
        run_parameters_file_path = os.path.join(default_params_directory, run_parameters_file)
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
            "store_date": True,
            "include_slope": False
        }

        run_eeek(args)

def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def create_control_file(data, output_filename, observation_count):
    with open(output_filename, "w") as file:
        file.write("pcf\n")  
        ## Control Data ##

        file.write("* control data\n")

        # Line 1
        line1 = data["control_data"]["line1"]
        file.write(f"{line1['RSTFLE']['value']} {line1['PESTMODE']['value']}\n")

        # Line 2
        line2 = data["control_data"]["line2"]
        file.write(
            f"{len(data["parameter_data"])} {observation_count} {len(data["parameter_data"])} {line2['NPRIOR']['value']} {line2['NOBSGP']['value']}\n"
        )

        # Line 3
        line3 = data["control_data"]["line3"]
        file.write(
            f"{line3['NTPLFLE']['value']} {line3['NINSFLE']['value']} {line3['PRECIS']['value']} {line3['DPOINT']['value']}\n"
        )

        # Line 4
        line4 = data["control_data"]["line4"]
        file.write(
            f"{line4['RLAMBDA1']['value']} {line4['RLAMFAC']['value']} {line4['PHIRATSUF']['value']} {line4['PHIREDLAM']['value']} {line4['NUMLAM']['value']}\n"
        )

        # Line 5
        line5 = data["control_data"]["line5"]
        file.write(
            f"{line5['RELPARMAX']['value']} {line5['FACPARMAX']['value']} {line5['ABSPARMAX']['value']}\n"
        )

        # Line 6
        line6 = data["control_data"]["line6"]
        file.write(
            f"{line6['PHIREDSWH']['value']}\n"
        )

        # Line 7
        line7 = data["control_data"]["line7"]
        file.write(
            f"{line7['NOPTMAX']['value']} {line7['PHIREDSTP']['value']} {line7['NPHISTP']['value']} {line7['NPHINORED']['value']} {line7['RELPARSTP']['value']} {line7['NRELPAR']['value']} {line7['PHISTOPTHRESH']['value']}\n"
        )

        # Line 8
        line8 = data["control_data"]["line8"]
        file.write(
            f"{line8['ICOV']['value']} {line8['ICOR']['value']} {line8['IEIG']['value']} {line8['IRES']['value']} {line8['JCOSAVE']['value']} {line8['JCOSAVEITN']['value']} {line8['VERBOSEREC']['value']} {line8['REISAVEITN']['value']} {line8['PARSAVEITN']['value']}\n"
        )

        # Singular Value Decompostion
        file.write("* singular value decomposition\n")
        svd = data["singular_value_decomposition"]

        file.write(f"{svd["line1"]['SVDMODE']['value']}\n")
        file.write(f"{svd["line2"]['MAXSING']['value']} {svd["line2"]['EIGTHRESH']['value']}\n")
        file.write(f"{svd["line3"]['EIGWRITE']['value']}\n")

        # Parameter Groups Section
        file.write("* parameter groups\n")
        for group in data["parameter_groups"]:
            file.write(f"{group['name']} {group['inctyp']} {group['derinc']} {group['derinclb']} {group['forcen']} {group['derincmul']} {group['splitthresh']}\n")
        
        # Parameter Data Section
        file.write("* parameter data\n")
        for param in data["parameter_data"]:
            file.write(f"{param['name']} {param['trans']} {param['inctyp']} {param['parval1']} {param['parlbnd']} {param['parubnd']} {param['pargp']} {param['scale']} {param['offset']}\n")

        # Observations groups
        file.write("* observation groups\n")
        file.write("obsgroup\n")

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
                    time = (pd.Timestamp(date, unit='ms') - pd.Timestamp('2016-01-01')).total_seconds() / (365.25 * 24 * 60 * 60)
                    phi = 6.283 * time
                    phi_cos = math.cos(phi)
                    phi_sin = math.sin(phi)
                    estimate = intercept + cos * phi_cos + sin * phi_sin

                    observations.append((f"intercept_{int(index)}_{observation_index}", intercept))
                    observations.append((f"cos_{int(index)}_{observation_index}", cos))
                    observations.append((f"sin_{int(index)}_{observation_index}", sin))
                    observations.append((f"estimate_{int(index)}_{observation_index}", estimate))
                    csv_writer.writerow([index, date, intercept, cos, sin, estimate])
                    observation_index += 1

            coefficients_2022 = dic["2022"]
            coefficients_2023 = dic["2023"]

            create_observation_from_coefficients(coefficients_2022["dates"], coefficients_2022["intercept"], coefficients_2022["cos"], coefficients_2022["sin"])
            create_observation_from_coefficients(coefficients_2023["dates"], coefficients_2023["intercept"], coefficients_2023["cos"], coefficients_2023["sin"])
            
        return observations

def write_observations_to_control_file(observations, file_path):
    with open(file_path, 'a') as file:
        file.write("* observation data\n")
        for observation_name, observation_value in observations:
            file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")

def create_pest_instruction_file(observations, instructions_filename):
    with open(instructions_filename, "w") as file:
        file.write("pif *\n")
        file.write(f"l1\n")
        grouped_observation = [observations[i:i+4] for i in range(0, len(observations), 4)]
        for obs in grouped_observation:
            intercept = obs[0][0]
            cos = obs[1][0]
            sin = obs[2][0]
            estimate = obs[3][0]

            file.write(f"l1 *,* !{intercept}! *,* !{cos}! *,* !{sin}! *,* !{estimate}!\n")

def append_model_and_io_sections(file_path):
    with open(file_path, 'a') as file:
        file.write("* model command line\n")
        file.write("model.bat\n")
        file.write("* model input/output\n")
        file.write("input.tpl  pest_input.csv\n")
        file.write("output.ins  pest_output.csv\n")

def create_template_file(template_filename):
    with open(template_filename, "w") as file:
        file.write("ptf #\n")
        file.write("#q1        #,0,0,0,#q5        #,0,0,0,#q9        #\n")
        file.write("#r         #\n")
        file.write("#p1        #,0,0,0,#p5        #,0,0,0,#p9        #\n")

def create_points_file(points_filename, coefficients_by_point):
    with open(points_filename, "w", newline="") as file:
        for idx, point in enumerate(coefficients_by_point):
            longitude = point["coordinates"][0]
            latitude = point["coordinates"][1]

            intercept = point["2022"]["intercept"]
            cos = point["2022"]["cos"]
            sin = point["2022"]["sin"]

            file.write(f"{longitude},{latitude},{intercept},{cos},{sin}\n")

def create_model_bat_file(file_path):
    with open(file_path, "w") as file:
        file.write(r"python C:\Users\kazad\OneDrive\Documents\GitHub\eeek\pest_eeek.py --input=pest_input.csv --output=pest_output.csv --points=points.csv --num_sinusoid_pairs=1 --include_intercept --store_measurement --collection=L8_L9_2022_2023 --store_estimate --store_date")

def get_dates_from_image_collection(year, coords):
    timestamps = [
        image["properties"]["millis"]
        for image in COLLECTIONS[f"L8_L9_2022_2023"].filterBounds(ee.Geometry.Point(coords)).getInfo()['features']
    ]

    return [
        timestamp for timestamp in timestamps if datetime.fromtimestamp(timestamp / 1000.0).year == year
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
        "dates": image_dates
    }

def fitted_coefficients_and_dates(points, fitted_coefficiets_filename):

    output_list = []
    coefficients_by_point = {}

    with open(fitted_coefficiets_filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["point", "longitude", "latitude", "intercept_2022", "cos_2022", "sin_2022", "intercept_2023", "cos_2023", "sin_2023"])
        for i, point in enumerate(points):
            coefficients_by_point[i] = {
                "coordinates": (point[0], point[1]),
                "2022": get_fitted_coefficients_for_point(COLLECTIONS["L8_L9_2022"].filterBounds(ee.Geometry.Point(point[0], point[1])), (point[0], point[1]), 2022),
                "2023": get_fitted_coefficients_for_point(COLLECTIONS["L8_L9_2023"].filterBounds(ee.Geometry.Point(point[0], point[1])), (point[0], point[1]), 2023)

            }

            csv_writer.writerow([i, point[0], point[1], coefficients_by_point[i]["2022"]["intercept"], coefficients_by_point[i]["2022"]["cos"], coefficients_by_point[i]["2022"]["sin"], coefficients_by_point[i]["2023"]["intercept"], coefficients_by_point[i]["2023"]["cos"], coefficients_by_point[i]["2023"]["sin"]])

            output_list.append(coefficients_by_point[i])
    
    return output_list

def harmonic_trend_coefficients(collection, coords):
    modality = {"constant": True, "linear": False, "unimodal": True, "bimodal": False, "trimodal": False}

    image_collection = ee.ImageCollection(collection.filterBounds(ee.Geometry.Point(coords)))

    reduced_image_collection_with_harmonics = add_harmonic_bands(image_collection, modality)

    harmonic_independent_variables = determine_harmonic_independents_via_modality_dictionary(modality)
    
    harmonic_one_time_regression = fit_harmonic_to_collection(reduced_image_collection_with_harmonics, "swir",harmonic_independent_variables)
    fitted_coefficients = harmonic_one_time_regression["harmonic_trend_coefficients"]

    return fitted_coefficients

def generate_measurements_and_target_fit_graphs(observations_filename, measurements_filename):
    output_directory = os.path.join(pest_run_directory, "measurements and target fit")

    delete_existing_directory_and_create_new(output_directory)

    observations = pd.read_csv(observations_filename)
    measurements = pd.read_csv(measurements_filename)

    observations["measurement"] = measurements["z"]

    observations["date"] = pd.to_datetime(observations['date'], unit='ms')

    grouped_observations = observations.groupby("point")

    for point, data in grouped_observations:
        fig, axs = plt.subplots(figsize=(12, 8))

        data = data[data["measurement"] != 0]
        
        axs.scatter(data['date'], data['measurement'], label='Measurement', color='red', s=10)
        axs.plot(data['date'], data['estimate'], label='Target Fit', color='green', linestyle='-')

        axs.xaxis.set_major_locator(mdates.AutoDateLocator())
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plot_filename = os.path.join(output_directory, f"point_{point}.png")
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(plot_filename)
        plt.close(fig)

if __name__ == "__main__":
    control_filename = pest_run_directory + "eeek.pst"
    instructions_filename = pest_run_directory + "output.ins"
    template_filename = pest_run_directory + "input.tpl"
    points_filename = pest_run_directory + "points.csv"
    model_filename = pest_run_directory + "model.bat"
    fitted_coefficiets_filename = pest_run_directory + "fitted_coefficients.csv"
    observations_filename = pest_run_directory + "observations.csv"

    parameters = read_json(parameters)
    points = read_json(points_coordinates)

    fitted_coefficiets_by_point = fitted_coefficients_and_dates(points['points'], fitted_coefficiets_filename)

    observations = build_observations(fitted_coefficiets_by_point, observations_filename)

    create_control_file(parameters, control_filename, len([x for x in observations if x[1] != 0]))
    write_observations_to_control_file(observations, control_filename)
    create_pest_instruction_file(observations, instructions_filename)
    append_model_and_io_sections(control_filename)
    create_template_file(template_filename)
    create_points_file(points_filename, fitted_coefficiets_by_point)
    create_model_bat_file(model_filename)

    run_eeek_with_default_parameters()

    generate_measurements_and_target_fit_graphs(observations_filename, f"{pest_run_directory}/default runs/default javascript/eeek_output.csv")

    print(f"Pest files has been created.")

