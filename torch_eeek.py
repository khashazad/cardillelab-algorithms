import torch
import pandas as pd
import multiprocessing as mp
from pprint import pprint
import math

NUM_MEASURES = 1  # eeek only supports one band at a time

band_names = [
    "point",
    "INTP",
    "COS0",
    "SIN0",
    "estimate",
    "z",
    "date",
    # "amplitude",
]


def sinusoidal(num_sinusoid_pairs, include_slope=True, include_intercept=True):
    """Creates sinusoid function of the form a+b*t+c*cos(2pi*t)+d*sin(2pi*t)...

    Useful for H in a Kalman filter setup.

    Args:
        num_sinusoid_pairs: int, number of sine + cosine terms in the model.
        include_slope: bool, if True include a linear slope term in the model.
        include_intercept: bool, if True include a bias/intercept term in the model.

    Returns:
        function that takes a torch.Tensor and returns a torch.Tensor
    """
    num_params = 0
    if include_intercept:
        num_params += 1
    if include_slope:
        num_params += 1
    num_params += 2 * num_sinusoid_pairs

    def sinusoidal_function(t):
        """Generates sinusoidal values based on the input time t.

        Args:
            t: torch.Tensor, time variable

        Returns:
            torch.Tensor
        """
        result = torch.zeros(num_params, dtype=torch.float32)
        idx = 0
        if include_intercept:
            result[idx] = 1.0  # Intercept term
            idx += 1
        if include_slope:
            result[idx] = t  # Slope term
            idx += 1
        for i in range(num_sinusoid_pairs):
            freq = (i + 1) * 2 * math.pi
            result[idx] = torch.cos(freq * t)
            result[idx + 1] = torch.sin(freq * t)
            idx += 2
        return result

    return sinusoidal_function


def kalman_filter(point_index, measurements, x0, P, F, Q, H, R, num_params):
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

    FREQUENCY = math.pi * 2

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
        x_bar = (F @ x).requires_grad_(True)
        P_bar = (F @ P @ F.t() + Q).requires_grad_(True)

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
        identity = torch.eye(num_params, requires_grad=True)

        # residual between measurement and prediction
        y = (z - (H @ x_bar)).requires_grad_(True)
        # covariance
        S = (((H @ P_bar) @ H.t()) + R).requires_grad_(True)
        S_inv = torch.inverse(S).requires_grad_(True)
        # Kalman gain: how much the prediction should be corrected by the measurement
        K = ((P_bar @ H.t()) @ S_inv).requires_grad_(True)
        # updated state
        x = (x_bar + (K @ y)).requires_grad_(True)
        # updated covariance
        P = ((identity - (K @ H)) @ P_bar).requires_grad_(True)

        return x, P

    states = [x0]
    covariances = [P]

    outputs = []

    for measurement, date_timestamp in measurements:
        x_prev = states[-1]
        P_prev = covariances[-1]

        # Convert timestamp to date and extract the year
        date = pd.to_datetime(date_timestamp, unit="ms")

        t = torch.tensor(
            (date - pd.to_datetime("2016-01-01")).total_seconds()
            / (365.25 * 24 * 60 * 60),
            dtype=torch.float32,
            requires_grad=True,
        )

        x_bar, P_bar = predict(x_prev, P_prev, F, Q)
        x, P = update(
            x_bar,
            P_bar,
            measurement,
            H(t).reshape(1, num_params),
            R,
            num_params,
        )

        if measurement == 0:
            x = x_prev
            P = P_prev

        states.append(x)
        covariances.append(P)

        intp = x[0]
        cos = x[1]
        sin = x[2]

        estimate = (
            intp + cos * torch.cos(t * FREQUENCY) + sin * torch.sin(t * FREQUENCY)
        ).requires_grad_(True)

        outputs.append(
            [
                point_index,
                x,
                estimate,
                measurement,
                date_timestamp,
            ]
        )

    return outputs


def main(args, Q, R):
    num_params = 3
    param_names = ["INTP", "COS0", "SIN0"]

    P = torch.tensor(
        [[0.00101, 0.0, 0.0], [0.0, 0.00222, 0.0], [0.0, 0.0, 0.00333]],
        dtype=torch.float32,
        requires_grad=True,
    )

    H = sinusoidal(
        args["num_sinusoid_pairs"],
        include_slope=args["include_slope"],
        include_intercept=args["include_intercept"],
    )

    kalman_init = {
        "F": torch.eye(num_params),
        "Q": Q,
        "H": H,
        "R": R,
        "num_params": num_params,
    }

    #################################################
    # Create parameters to run filter on each point #
    #################################################
    points = []
    with open(args["points"], "r") as f:
        for i, line in enumerate(f.readlines()):
            lon, lat, x1, x2, x3 = line.split(",")
            points.append(
                {
                    "index": i,
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "x0": [float(x1), float(x2), float(x3)],
                }
            )

    all_measurements = pd.read_csv(args["measurements"])

    #################################################
    ##### Run Kalman filter across all points #######
    #################################################
    def process_point(kwargs):
        index = kwargs["index"]

        measurements_for_point = all_measurements[
            all_measurements["point"] == kwargs["index"]
        ]
        measurements = measurements_for_point[["swir", "date"]].values.tolist()

        x0 = torch.tensor(
            kwargs["x0"], dtype=torch.float32, requires_grad=True
        ).reshape(num_params, NUM_MEASURES)

        kalman_result = kalman_filter(
            index,
            measurements,
            x0,
            P,
            **kalman_init,
        )

        return kalman_result

    all_results = []
    for p in points:
        all_results.extend(process_point(p))

    df = pd.DataFrame(
        [
            [
                index,
                x[0].item(),
                x[1].item(),
                x[2].item(),
                estimate,
                measurement,
                date_timestamp,
            ]
            for index, x, estimate, measurement, date_timestamp in all_results
        ],
        columns=band_names,
    )

    df.to_csv(args["output"], index=False)

    return [
        [
            row[1][0],
            row[1][1],
            row[1][2],
        ]
        for row in all_results
    ]
