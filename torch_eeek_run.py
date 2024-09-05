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


class RunType(Enum):
    SINGLE_RUN = "single run"
    OPTIMIZATION = "optimization"


POINT_SET = 7
RUN_VERSION = 1
RUN_TYPE = RunType.OPTIMIZATION


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


class KalmanFilterModel(nn.Module):
    def __init__(self):
        super(KalmanFilterModel, self).__init__()

    def forward(self, run_directory):
        args = {
            "measurements": f"{run_directory}/measurements.csv",
            "points": f"{run_directory}/points.csv",
            "output": f"{run_directory}/eeek_output.csv",
            "parameters_output": f"{run_directory}/eeek_parameters.csv",
            "include_intercept": True,
            "include_slope": False,
            "num_sinusoid_pairs": 1,
        }

        output = run_eeek(args, self.Q, self.R)

        return output


# Define the loss function to handle a list of tuples (3 values per tuple)
def loss_function(estimates, true_states):
    # Convert the list of tuples to a tensor for easier computation
    estimates_tensor = estimates
    true_states_tensor = true_states

    return torch.mean((estimates_tensor - true_states_tensor) ** 2)


def optimize_parameters(run_directory, q1, q5, q9, r, epochs=500):
    kf_model = KalmanFilterModel()

    kf_model.Q = nn.Parameter(
        torch.tensor(
            [[q1, 0.0, 0.0], [0.0, q5, 0.0], [0.0, 0.0, q9]], requires_grad=True
        )
    )
    kf_model.R = nn.Parameter(torch.tensor(r, requires_grad=True))

    # # Example optimizer (Adam)
    optimizer = optim.Adam(kf_model.parameters(), lr=0.01)

    true_states = torch.tensor(
        np.loadtxt(
            f"{run_directory}/observations.csv",
            delimiter=",",
            skiprows=1,
            usecols=(2, 3, 4),
        ),
        requires_grad=True,
    )

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass: get the estimate updates from Kalman filter
        estimates = kf_model(run_directory)

        # Calculate loss (compare estimates with ground truth)
        loss = loss_function(estimates, true_states)

        # Backward pass: compute gradients
        loss.backward()

        # print(kf_model.Q1.grad, kf_model.Q5.grad, kf_model.Q9.grad, kf_model.R.grad)
        # print(kf_model.Q.grad, kf_model.R.grad)

        # Update the parameters (Q and R)
        optimizer.step()

        # Print the loss for monitoring
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # print(
        #     f"Q1: {kf_model.Q1.item()}, Q5: {kf_model.Q5.item()}, Q9: {kf_model.Q9.item()}, R: {kf_model.R.item()}"
        # )

    # After training, check the optimized Q1, Q2, Q3, R
    print(
        f"Optimized Q1: {kf_model.Q[0][0].item()}, Q5: {kf_model.Q[1][1].item()}, Q9: {kf_model.Q[2][2].item()}, R: {kf_model.R.item()}"
    )


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
