import os
from eeek.gather_collections import reduce_collection_to_points_and_write_to_file
from eeek.image_collections import COLLECTIONS
from eeek.prepare_optimization_run import (
    build_observations,
    fitted_coefficients_and_dates,
    parse_point_coordinates,
    create_points_file,
)
from torch_eeek import main as eeek
from utils.analysis import analyze_results
from enum import Enum
from caliberate_parameters_pytorch import optimize_parameters
import torch


class RunType(Enum):
    SINGLE_RUN = "single run"
    OPTIMIZATION = "optimization"


POINT_SET = 10
RUN_VERSION = 2
RUN_TYPE = RunType.SINGLE_RUN


initial_params = {"q1": 0.00125, "q5": 0.000125, "q9": 0.000125, "r": 0.003}

graph_flags = {
    "estimate": True,
    "final_2022_fit": False,
    "final_2023_fit": False,
    "intercept_cos_sin": True,
    "residuals": True,
    "amplitude": True,
}

root = os.path.dirname(os.path.abspath(__file__))

run_directory = os.path.join(
    root, "torch runs", f"point set {POINT_SET}", f"v{RUN_VERSION}"
)

os.makedirs(run_directory, exist_ok=True)

if __name__ == "__main__":
    points = parse_point_coordinates(f"{root}/points/sets/{POINT_SET}")

    if not os.path.exists(f"{run_directory}/measurements.csv"):
        reduce_collection_to_points_and_write_to_file(
            COLLECTIONS["L8_L9_2022_2023"], points, f"{run_directory}/measurements.csv"
        )

    if not os.path.exists(f"{run_directory}/fitted_coefficients.csv"):
        fitted_coefficiets_by_point = fitted_coefficients_and_dates(
            points, f"{run_directory}/fitted_coefficients.csv"
        )

        create_points_file(f"{run_directory}/points.csv", fitted_coefficiets_by_point)

        observations = build_observations(
            fitted_coefficiets_by_point, f"{run_directory}/observations.csv"
        )

    if RUN_TYPE == RunType.SINGLE_RUN:
        args = {
            "measurements": f"{run_directory}/measurements.csv",
            "points": f"{run_directory}/points.csv",
            "output": f"{run_directory}/eeek_output.csv",
            "parameters_output": f"{run_directory}/eeek_parameters.csv",
            "include_intercept": True,
            "include_slope": False,
            "num_sinusoid_pairs": 1,
        }

        eeek(
            args,
            torch.tensor(
                [
                    [initial_params["q1"], 0, 0],
                    [0, initial_params["q5"], 0],
                    [0, 0, initial_params["q9"]],
                ]
            ),
            torch.tensor([initial_params["r"]]),
        )
    else:
        optimize_parameters(run_directory, **initial_params)

    analyze_results(run_directory, graph_flags)
