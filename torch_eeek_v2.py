import torch
import pandas as pd
import multiprocessing as mp
from pprint import pprint
import math


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


def kalman_filter(
    point_index,
    measurements,
    x0,
    adaptive_threshold,
    adaptive_scale_factor,
    P,
    F,
    Q,
    H,
    R,
    num_params,
):
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

        # residual between measurement and prediction
        y = z - (H @ x_bar)
        # measurement prediction covariance
        S = ((H @ P_bar) @ H.t()) + R
        S_inv = torch.inverse(S)
        # Kalman gain: how much the state prediction should be corrected by the measurement
        K = (P_bar @ H.t()) @ S_inv
        # updated state
        x = x_bar + (K @ y)
        # updated covariance
        P = (identity - (K @ H)) @ P_bar

        return x, P

    states = torch.empty((0, 3), requires_grad=True)
    states = torch.cat([states, x0.flatten().unsqueeze(0)], dim=0)

    covariances = torch.empty((0, 9), requires_grad=True)
    covariances = torch.cat([covariances, P.flatten().unsqueeze(0)], dim=0)

    results = torch.empty((0, 7), requires_grad=True)

    for measurement, date_timestamp in measurements:
        x_prev = states[-1].reshape(num_params, 1)
        P_prev = covariances[-1].reshape(num_params, num_params)

        # Convert timestamp to date and extract the year
        date = pd.to_datetime(date_timestamp, unit="ms")

        t = torch.tensor(
            (date - pd.to_datetime("2016-01-01")).total_seconds()
            / (365.25 * 24 * 60 * 60),
            dtype=torch.float32,
        )

        estimate_diff = measurement - (H(t) @ x_prev)
        Q_adapted = adaptive_process_noise(Q, estimate_diff, adaptive_threshold, adaptive_scale_factor)

        x_bar, P_bar = predict(x_prev, P_prev, F, Q_adapted)
        x, P = update(
            x_bar,
            P_bar,
            measurement,
            H(t).reshape(1, num_params),
            R,
            num_params,
        )

        if abs(measurement) < 1e-30:
            x = x_prev
            P = P_prev

        states = torch.cat([states, x.flatten().unsqueeze(0)], dim=0)
        covariances = torch.cat([covariances, P.flatten().unsqueeze(0)], dim=0)

        intp = x[0]
        cos = x[1]
        sin = x[2]

        estimate = (
            intp + cos * torch.cos(t * FREQUENCY) + sin * torch.sin(t * FREQUENCY)
        )

        results = torch.cat(
            [
                results,
                torch.stack(
                    [
                        torch.tensor(point_index, dtype=torch.int).unsqueeze(0),
                        intp.unsqueeze(0) if intp.dim() == 0 else intp,
                        cos.unsqueeze(0) if cos.dim() == 0 else cos,
                        sin.unsqueeze(0) if sin.dim() == 0 else sin,
                        estimate.unsqueeze(0) if estimate.dim() == 0 else estimate,
                        torch.tensor(measurement, dtype=torch.float32).unsqueeze(0),
                        torch.tensor(date_timestamp, dtype=torch.int64).unsqueeze(0),
                    ],
                )
                .flatten()
                .unsqueeze(0),
            ],
            dim=0,
        )

    return results


def main(args, Q, R, P, adaptive_threshold, adaptive_scale_factor):
    num_params = 3

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
                    "x0": torch.tensor(
                        [[float(x1)], [float(x2)], [float(x3)]],
                        dtype=torch.float32,
                    ),
                }
            )

    all_measurements = pd.read_csv(args["measurements"])

    #################################################
    ##### Run Kalman filter across all points #######
    #################################################
    def process_point(kwargs):
        index = kwargs["index"]

        measurements = all_measurements[all_measurements["point"] == kwargs["index"]][
            ["swir", "date"]
        ].values.tolist()

        return kalman_filter(
            index,
            measurements,
            kwargs["x0"],
            adaptive_threshold,
            adaptive_scale_factor,
            P,
            **kalman_init,
        )

    all_results = torch.empty(0, requires_grad=True)

    for p in points:
        results = process_point(p)
        all_results = torch.cat([all_results, results])

    outputs_df = pd.DataFrame(all_results.detach().numpy(), columns=band_names)
    outputs_df["point"] = outputs_df["point"].astype(int)
    outputs_df.to_csv(args["output"], index=False)

    return all_results


def adaptive_process_noise(Q_initial, estimate_diff, threshold=0.1, scale_factor=10):
    """
    Adjusts process noise Q matrix based on the magnitude of the difference
    between successive estimates. This helps the filter be stable when measurements
    are similar, but react when changes occur.

    Args:
        Q_initial (torch.Tensor): Initial process noise covariance matrix.
        estimate_diff (torch.Tensor): Difference between current measurement and state estimate.
        threshold (float): Difference threshold below which Q is decreased.
        scale_factor (float): Factor by which to scale Q when significant change is detected.

    Returns:
        torch.Tensor: Adapted process noise covariance matrix.
    """

    # Compute the norm of the estimate difference to quantify the change
    diff_norm = torch.norm(estimate_diff)

    if diff_norm < threshold:
        # Reduce Q when changes are small (stabilize coefficients)
        Q_adapted = Q_initial / scale_factor
    else:
        # Increase Q when changes are significant (allow adaptation)
        Q_adapted = Q_initial * scale_factor

    return Q_adapted
