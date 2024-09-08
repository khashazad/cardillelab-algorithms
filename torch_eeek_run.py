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
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch_eeek import main as run_eeek
import numpy as np
import json
from utils.charts import generate_charts_comparing_runs
from pprint import pprint
import torch.multiprocessing as mp


class RunType(Enum):
    SINGLE_RUN = "single run"
    OPTIMIZATION = "optimization"


POINT_SET = 8
RUN_TYPE = RunType.OPTIMIZATION
LOOPS = 1000


param_sets = [
    {
        "q1": 0.00125,
        "q5": 0.000125,
        "q9": 0.000125,
        "r": 0.003,
    },
    {
        "q1": 0.00070531606,
        "q5": 0.00026831817,
        "q9": 0.00417308787,
        "r": 0.00124919674,
    },
]

root = os.path.dirname(os.path.abspath(__file__))

parent_run_directory = os.path.join(root, "torch runs", f"point set {POINT_SET}")

os.makedirs(parent_run_directory, exist_ok=True)


class KalmanFilterModel(nn.Module):
    def __init__(self):
        super(KalmanFilterModel, self).__init__()

    def forward(self, parent_run_directory, run_directory):
        args = {
            "measurements": f"{parent_run_directory}/measurements.csv",
            "points": f"{parent_run_directory}/points.csv",
            "output": f"{run_directory}/eeek_output.csv",
            "parameters_output": f"{run_directory}/eeek_parameters.csv",
            "include_intercept": True,
            "include_slope": False,
            "num_sinusoid_pairs": 1,
        }

        output = run_eeek(
            args,
            torch.diag(torch.stack([self.Q1, self.Q5, self.Q9])),
            self.R,
        )

        return output


def loss_function(estimates, true_states):
    estimates_tensor = estimates
    true_states_tensor = true_states

    return torch.mean((estimates_tensor - true_states_tensor) ** 2)


def optimize_parameters(
    parent_run_directory, run_directory, q1, q5, q9, r, max_iteration=LOOPS
):
    kf_model = KalmanFilterModel()

    kf_model.Q1 = nn.Parameter(
        torch.tensor(
            q1,
            dtype=torch.float32,
            requires_grad=True,
        )
    )
    kf_model.Q5 = nn.Parameter(
        torch.tensor(
            q5,
            dtype=torch.float32,
            requires_grad=True,
        )
    )
    kf_model.Q9 = nn.Parameter(
        torch.tensor(
            q9,
            dtype=torch.float32,
            requires_grad=True,
        )
    )

    kf_model.R = nn.Parameter(torch.tensor(r, requires_grad=True))

    optimizer = optim.Adam(
        [
            {"params": kf_model.Q1, "lr": 0.005},
            {"params": kf_model.Q5, "lr": 0.005},
            {"params": kf_model.Q9, "lr": 0.005},
            {"params": kf_model.R, "lr": 0.005},
        ]
    )

    true_states = torch.tensor(
        np.loadtxt(
            f"{parent_run_directory}/observations.csv",
            delimiter=",",
            skiprows=1,
            usecols=(2, 3, 4),
        ),
        requires_grad=True,
    )

    # Training loop
    for iteration in range(max_iteration):
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass: get the estimate updates from Kalman filter
        estimates = kf_model(parent_run_directory, run_directory)

        # Calculate loss (compare estimates with ground truth)
        loss = loss_function(estimates, true_states)

        # Backward pass: compute gradients
        loss.backward()

        # with torch.no_grad():
        #     p.data = torch.clamp(p.data, min=1e-50)
        #     kf_model.Q5.data = torch.clamp(kf_model.Q5.data, min=1e-50)
        #     kf_model.Q9.data = torch.clamp(kf_model.Q9.data, min=1e-50)
        #     kf_model.R.data = torch.clamp(kf_model.R.data, min=1e-50)

        # print(
        #     f"Gradients: Q1: {kf_model.Q1.grad}, Q5: {kf_model.Q5.grad}, Q9: {kf_model.Q9.grad}, R: {kf_model.R.grad}"
        # )

        optimizer.step()

        for p in kf_model.parameters():
            p.data = p.data.clamp(1e-50, 1)
        # print(f"Epoch {iteration+1}/{max_iteration}, Loss: {loss.item()}")

        if iteration == 0:
            prev_loss = loss.item()
            no_change_count = 0
        else:
            if abs(prev_loss - loss.item()) < 1e-10:
                no_change_count += 1
            else:
                no_change_count = 0

            prev_loss = loss.item()

            if no_change_count >= 5:
                print(
                    f"Early stopping at iteration {iteration+1} due to no significant change in loss."
                )
                break

    optimized_params = {
        "optimized_Q1": kf_model.Q1.item(),
        "optimized_Q5": kf_model.Q5.item(),
        "optimized_Q9": kf_model.Q9.item(),
        "optimized_R": kf_model.R.item(),
        "final_loss": loss.item(),
    }

    with open(f"{run_directory}/eeek_params.json", "r+") as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError:
            data = {}
        data.update(optimized_params)
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()


def run_torch_eeek(parent_run_directory, param_set):
    run_version = (
        len([f for f in os.listdir(parent_run_directory) if f.startswith("v")]) + 1
    )
    run_directory = os.path.join(parent_run_directory, f"v{run_version}")
    os.makedirs(run_directory, exist_ok=True)

    with open(f"{run_directory}/eeek_params.json", "w") as json_file:
        json.dump(param_set, json_file, indent=4)

    if RUN_TYPE == RunType.SINGLE_RUN:
        args = {
            "measurements": f"{parent_run_directory}/measurements.csv",
            "points": f"{parent_run_directory}/points.csv",
            "output": f"{parent_run_directory}/eeek_output.csv",
            "parameters_output": f"{parent_run_directory}/eeek_parameters.csv",
            "include_intercept": True,
            "include_slope": False,
            "num_sinusoid_pairs": 1,
        }

        eeek(
            args,
            torch.diag(
                torch.tensor(
                    [param_set["q1"], param_set["q5"], param_set["q9"]],
                    dtype=torch.float32,
                )
            ),
            torch.tensor(param_set["r"]),
        )
    else:
        optimize_parameters(parent_run_directory, run_directory, **param_set)

    generate_charts_comparing_runs(
        {"run": f"{run_directory}/eeek_output.csv"},
        f"{parent_run_directory}/observations.csv",
        f"{parent_run_directory}/analysis/v{run_version}",
        {
            "estimate": True,
            "final_2022_fit": False,
            "final_2023_fit": False,
            "intercept_cos_sin": True,
            "residuals": True,
            "amplitude": True,
        },
    )


def run_torch_eeek_wrapper(param_set):
    run_torch_eeek(parent_run_directory, param_set)


if __name__ == "__main__":
    if not os.path.exists(f"{parent_run_directory}/measurements.csv"):
        points = parse_point_coordinates(f"{root}/points/sets/{POINT_SET}")
        reduce_collection_to_points_and_write_to_file(
            COLLECTIONS["L8_L9_2022_2023"],
            points,
            f"{parent_run_directory}/measurements.csv",
        )
        fitted_coefficiets_by_point = fitted_coefficients_and_dates(
            points, f"{parent_run_directory}/fitted_coefficients.csv"
        )

        create_points_file(
            f"{parent_run_directory}/points.csv", fitted_coefficiets_by_point
        )

        observations = build_observations(
            fitted_coefficiets_by_point, f"{parent_run_directory}/observations.csv"
        )

    mp.set_start_method("spawn")
    with mp.Pool(processes=len(param_sets)) as pool:
        pool.map(run_torch_eeek_wrapper, param_sets)
