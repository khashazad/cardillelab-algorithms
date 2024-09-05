import pandas as pd
import torch
from datetime import datetime

FREQUENCY = 6.283


def predict(x, P, F, Q):
    """Performs the predict step of the Kalman Filter loop.

    Args:
        x: torch.Tensor (n x 1), the state
        P: torch.Tensor (n x n), the state covariance
        F: torch.Tensor (n x n), the process model
        Q: torch.Tensor (n x n), the process noise

    Returns:
        x_bar, P_bar: the predicted state and the predicted state covariance.
    """
    x_bar = F @ x
    P_bar = F @ P @ F.t() + Q

    return x_bar, P_bar


def update(x_bar, P_bar, z, H, R, num_params):
    """Performs the update step of the Kalman Filter loop.

    Args:
        x_bar: torch.Tensor (n x 1), the predicted state
        P_bar: torch.Tensor (n x n), the predicted state covariance
        z: torch.Tensor (1 x 1), the measurement
        H: torch.Tensor (1 x n), the measurement function
        R: torch.Tensor (1 x 1), the measurement noise
        num_params: int, the number of parameters in the state variable

    Returns:
        x, P: the updated state and state covariance
    """
    identity = torch.eye(num_params)

    H = H.reshape(1, num_params)
    y = z - (H @ x_bar)
    S = ((H @ P_bar) @ H.t()) + R
    S_inv = torch.inverse(S)
    K = (P_bar @ H.t()) @ S_inv
    x = x_bar + (K @ y)
    P = (identity - (K @ H)) @ P_bar

    return x, P


def kalman_filter(measurements, x0, P, F, Q, H, R, num_params):
    """Applies a Kalman Filter to the given data.

    Args:
        data: list of measurements (torch.Tensor)
        init_state: torch.Tensor, the initial state
        init_covariance: torch.Tensor, the initial state covariance
        F: torch.Tensor, the process model
        Q: torch.Tensor, the process noise
        H: torch.Tensor, the measurement function
        R: torch.Tensor, the measurement noise
        num_params: int, number of parameters in the state

    Returns:
        list of states and covariances after applying Kalman Filter
    """
    states = [x0]
    covariances = [P]

    outputs = []

    for measurement, date in measurements:
        x_prev = states[-1]
        P_prev = covariances[-1]

        # Convert timestamp to date and extract the year
        date = pd.to_datetime(date, unit="ms")

        t = torch.tensor(
            (date - pd.to_datetime("2016-01-01")).total_seconds()
            / (365.25 * 24 * 60 * 60),
            dtype=torch.float32,
        )

        x_bar, P_bar = predict(x_prev, P_prev, F, Q)
        x, P = update(x_bar, P_bar, measurement, H(t), R, num_params)

        if measurement is None:
            x = x_prev
            P = P_prev

        intp = x[0]
        cos = x[1]
        sin = x[2]

        estimate = (
            intp + cos * torch.cos(t * FREQUENCY) + sin * torch.sin(t * FREQUENCY)
        )
        amplitude = torch.sqrt(cos**2 + sin**2)

        outputs.append(
            [
                measurement,
                x,
                estimate,
                date,
                # amplitude,
            ]
        )

    return outputs
