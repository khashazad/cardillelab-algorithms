from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from torch_eeek import main as run_eeek
import os
import numpy as np


VERSION = 1

Q1 = 0.00125
Q5 = 0.00125
Q9 = 0.00125
R = 0.003


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

        return torch.tensor(run_eeek(args, self.Q1, self.Q5, self.Q9, self.R))


# Define the loss function to handle a list of tuples (3 values per tuple)
def loss_function(estimates, true_states):
    # Convert the list of tuples to a tensor for easier computation
    estimates_tensor = torch.tensor(
        estimates, requires_grad=True
    )  # Shape: [N, 3] where N is the number of time steps
    true_states_tensor = torch.tensor(
        true_states, requires_grad=True
    )  # Ground truth should be in the same format

    return torch.mean((estimates_tensor - true_states_tensor) ** 2)


def optimize_parameters(run_directory, q1, q5, q9, r, epochs=30):
    kf_model = KalmanFilterModel()

    kf_model.Q1 = nn.Parameter(torch.tensor(q1, requires_grad=True))
    kf_model.Q5 = nn.Parameter(torch.tensor(q5, requires_grad=True))
    kf_model.Q9 = nn.Parameter(torch.tensor(q9, requires_grad=True))
    kf_model.R = nn.Parameter(torch.tensor(r, requires_grad=True))

    # # Example optimizer (Adam)
    optimizer = optim.Adam(kf_model.parameters(), lr=1)

    true_states = np.loadtxt(
        f"{run_directory}/observations.csv",
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 4),
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

        print(kf_model.Q1.grad, kf_model.Q5.grad, kf_model.Q9.grad, kf_model.R.grad)

        # Update the parameters (Q and R)
        optimizer.step()

        # Print the loss for monitoring
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        print(
            f"Q1: {kf_model.Q1.item()}, Q5: {kf_model.Q5.item()}, Q9: {kf_model.Q9.item()}, R: {kf_model.R.item()}"
        )

    # After training, check the optimized Q1, Q2, Q3, R
    print(
        f"Optimized Q1: {kf_model.Q1.item()}, Q2: {kf_model.Q5.item()}, Q3: {kf_model.Q9.item()}, R: {kf_model.R.item()}"
    )
