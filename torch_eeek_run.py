import os
from utils.ee.gather_collections import reduce_collection_to_points_and_write_to_file
from lib.image_collections import COLLECTIONS
from utils.prepare_optimization_run import (
    build_observations,
    fitted_coefficients_and_dates,
    parse_point_coordinates,
    create_points_file,
)
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch.torch_eeek import main as run_eeek
import numpy as np
import json
from utils.visualization.charts import generate_charts_single_run
from pprint import pprint
import torch.multiprocessing as mp


class RunType(Enum):
    SINGLE_RUN = "single run"
    OPTIMIZATION = "optimization"


class ObservationType(Enum):
    STATE = "state"
    ESTIMATE = "estimate"
    ALL = "all"


POINT_SET = 10
RUN_TYPE = RunType.SINGLE_RUN
OBSERVATION_TYPE = ObservationType.ALL
ITERATIONS = 100
CONDITIONAL_Q = False


param_sets = [
    {
        "q1": 0.00125,
        "q5": 0.000125,
        "q9": 0.000125,
        "r": 0.003,
        "p1": 0.00101,
        "p5": 0.00222,
        "p9": 0.00333,
        "adaptive_threshold": 0.1,
        "adaptive_scale_factor": 10.0,
    }
]

parameter_learning_rates = {
    "q1": {
        "lr": 0.001,
        "momentum": 0.85,
    },
    "q5": {
        "lr": 0.001,
        "momentum": 0.85,
    },
    "q9": {
        "lr": 0.001,
        "momentum": 0.85,
    },
    "r": {
        "lr": 0.001,
        "momentum": 0.9,
    },
    "p1": {
        "lr": 0.001,
        "momentum": 0.9,
    },
    "p5": {
        "lr": 0.001,
        "momentum": 0.9,
    },
    "p9": {
        "lr": 0.001,
        "momentum": 0.9,
    },
    "adaptive_threshold": None,
    "adaptive_scale_factor": None,
}

root = os.path.dirname(os.path.abspath(__file__))

parent_run_directory = os.path.join(
    root, "pytorch", "torch runs", f"point set {POINT_SET}"
)

os.makedirs(parent_run_directory, exist_ok=True)


def get_paramters_to_optimize(model):
    parameters_to_optimize = []

    if parameter_learning_rates["q1"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.Q1,
                "lr": parameter_learning_rates["q1"]["lr"],
                "momentum": parameter_learning_rates["q1"]["momentum"],
            }
        )
    if parameter_learning_rates["q5"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.Q5,
                "lr": parameter_learning_rates["q5"]["lr"],
                "momentum": parameter_learning_rates["q5"]["momentum"],
            }
        )
    if parameter_learning_rates["q9"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.Q9,
                "lr": parameter_learning_rates["q9"]["lr"],
                "momentum": parameter_learning_rates["q9"]["momentum"],
            }
        )
    if parameter_learning_rates["r"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.R,
                "lr": parameter_learning_rates["r"]["lr"],
                "momentum": parameter_learning_rates["r"]["momentum"],
            }
        )
    if parameter_learning_rates["p1"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.P1,
                "lr": parameter_learning_rates["p1"]["lr"],
                "momentum": parameter_learning_rates["p1"]["momentum"],
            }
        )
    if parameter_learning_rates["p5"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.P5,
                "lr": parameter_learning_rates["p5"]["lr"],
                "momentum": parameter_learning_rates["p5"]["momentum"],
            }
        )
    if parameter_learning_rates["p9"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.P9,
                "lr": parameter_learning_rates["p9"]["lr"],
                "momentum": parameter_learning_rates["p9"]["momentum"],
            }
        )

    if parameter_learning_rates["adaptive_threshold"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.adaptive_threshold,
                "lr": parameter_learning_rates["adaptive_threshold"]["lr"],
                "momentum": parameter_learning_rates["adaptive_threshold"]["momentum"],
            }
        )

    if parameter_learning_rates["adaptive_scale_factor"] is not None:
        parameters_to_optimize.append(
            {
                "params": model.adaptive_scale_factor,
                "lr": parameter_learning_rates["adaptive_scale_factor"]["lr"],
                "momentum": parameter_learning_rates["adaptive_scale_factor"][
                    "momentum"
                ],
            }
        )

    return parameters_to_optimize


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

        return run_eeek(
            args,
            torch.diag(torch.stack([self.Q1, self.Q5, self.Q9])),
            self.R,
            torch.diag(torch.stack([self.P1, self.P5, self.P9])),
            None,
            None,
            # self.adaptive_threshold,
            # self.adaptive_scale_factor,
        )


def loss_function(estimates, target):
    if OBSERVATION_TYPE == ObservationType.ALL:
        estimates = estimates[:, 1:5]
    if OBSERVATION_TYPE == ObservationType.ESTIMATE:
        estimates = estimates[:, 4]
    if OBSERVATION_TYPE == ObservationType.STATE:
        estimates = estimates[:, 1:4]

    return torch.mean((estimates - target) ** 2)


def optimize_parameters(
    parent_run_directory,
    run_directory,
    q1,
    q5,
    q9,
    r,
    p1,
    p5,
    p9,
    adaptive_threshold,
    adaptive_scale_factor,
):
    kf_model = KalmanFilterModel()

    kf_model.Q1 = nn.Parameter(
        torch.tensor(q1, dtype=torch.float32, requires_grad=True)
    )
    kf_model.Q5 = nn.Parameter(
        torch.tensor(q5, dtype=torch.float32, requires_grad=True)
    )
    kf_model.Q9 = nn.Parameter(
        torch.tensor(q9, dtype=torch.float32, requires_grad=True)
    )

    kf_model.R = nn.Parameter(torch.tensor(r, requires_grad=True))

    kf_model.P1 = nn.Parameter(torch.tensor(p1, requires_grad=True))
    kf_model.P5 = nn.Parameter(torch.tensor(p5, requires_grad=True))
    kf_model.P9 = nn.Parameter(torch.tensor(p9, requires_grad=True))

    if CONDITIONAL_Q:
        kf_model.adaptive_threshold = nn.Parameter(
            torch.tensor(adaptive_threshold, requires_grad=True)
        )
        kf_model.adaptive_scale_factor = nn.Parameter(
            torch.tensor(adaptive_scale_factor, requires_grad=True)
        )

    optimizer = optim.Adam(get_paramters_to_optimize(kf_model))

    if OBSERVATION_TYPE == ObservationType.ALL:
        columns = (2, 3, 4, 5)
    elif OBSERVATION_TYPE == ObservationType.STATE:
        columns = (2, 3, 4)
    elif OBSERVATION_TYPE == ObservationType.ESTIMATE:
        columns = (5,)

    true_states = torch.tensor(
        np.loadtxt(
            f"{parent_run_directory}/observations.csv",
            delimiter=",",
            skiprows=1,
            usecols=columns,
        ),
        requires_grad=True,
    )

    # Training loop
    for iteration in range(ITERATIONS):
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass: get the estimate updates from Kalman filter
        estimates = kf_model(parent_run_directory, run_directory)

        # Calculate loss (compare observations with target)
        loss = loss_function(estimates, true_states)

        if RUN_TYPE == RunType.SINGLE_RUN:
            break

        # Backward pass: compute gradients
        loss.backward()

        # print(
        #     f"Gradients: Q1: {kf_model.Q1.grad}, Q5: {kf_model.Q5.grad}, Q9: {kf_model.Q9.grad}, R: {kf_model.R.grad}"
        # )

        optimizer.step()

        for p in kf_model.parameters():
            if p in [kf_model.Q1, kf_model.Q5, kf_model.Q9, kf_model.R]:
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

            if no_change_count >= 10:
                print(
                    f"Early stopping at iteration {iteration+1} due to no significant change in loss."
                )
                break

    optimized_params = {
        "optimized_Q1": kf_model.Q1.item(),
        "optimized_Q5": kf_model.Q5.item(),
        "optimized_Q9": kf_model.Q9.item(),
        "optimized_R": kf_model.R.item(),
        "optimized_P1": kf_model.P1.item(),
        "optimized_P5": kf_model.P5.item(),
        "optimized_P9": kf_model.P9.item(),
        # "optimized_adaptive_threshold": kf_model.adaptive_threshold.item(),
        # "optimized_adaptive_scale_factor": kf_model.adaptive_scale_factor.item(),
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

    optimize_parameters(parent_run_directory, run_directory, **param_set)

    generate_charts_single_run(
        f"{run_directory}/eeek_output.csv",
        f"{parent_run_directory}/observations.csv",
        f"{run_directory}/analysis",
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
