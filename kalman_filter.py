import math
import string

import ee
import numpy as np

# band names
STATE = "x"
COV = "P"
MEASUREMENT = "z"
ZPRIME = "zprime"

UNMASK_VALUE = 0

# use 2pi * time since this date as input to sinusoids
START_DATE = "2016-01-01"


def predict(x, P, F, Q):
    """Performs the predict step of the Kalman Filter loop.

    Args:
        x: ee array Image (n x 1), the state
        P: ee array Image (n x n), the state covariance
        F: ee array_image (n x n), the process model
        Q: ee array Image (n x n), the process noise

    Returns:
        x_bar (ee array image), P_bar (ee array image): the predicted state,
        and the predicted state covariance.
    """
    x_bar = F.matrixMultiply(x)
    P_bar = F.matrixMultiply(P).matrixMultiply(F.matrixTranspose()).add(Q)
    return x_bar, P_bar


def update(x_bar, P_bar, z, H, R, I):
    """Performs the update step of the Kalman Filter loop.

    Args:
        x_bar: ee array image (n x 1), the predicted state
        P_bar: ee array image (n x n), the predicted state covariance
        z: ee array Image (1 x 1), the measurement
        H: ee array image (1 x n), the measurement function
        R: ee array Image (1 x 1), the measurement noise
        I: ee array Image (n x n), the identity matrix of the proper size

    Returns:
        x (ee array image), P (ee array image): the updated state and state
        covariance
    """
    y = z.subtract(H.matrixMultiply(x_bar))
    S = H.matrixMultiply(P_bar).matrixMultiply(H.matrixTranspose()).add(R)
    S_inv = S.matrixInverse()
    K = P_bar.matrixMultiply(H.matrixTranspose()).matrixMultiply(S_inv)
    x = x_bar.add(K.matrixMultiply(y))
    P = (I.subtract(K.matrixMultiply(H))).matrixMultiply(P_bar)
    return x, P


def kalman_filter(collection, init_x, init_P, F, Q, H, R, num_params=3):
    """Applies a Kalman Filter to the given image collection.

    In case they need it, F, Q, H, and R are all given x, P, z, and t as named
    inputs and are reevaluated at each step of the Kalman Filter loop. In case
    they don't need some/all of those set **kwargs as a parameter. E.g., if H
    relies only on t, define it as `def H(t, **kwargs): ...` and if Q does not
    need any inputs, define it as `def Q(**kwargs): ...`

    Args:
        collection: ee.ImageCollection, used as the measurements
        init_x, ee.Image, guess of initial state
        init_P, ee.Image, guess of initial state covariance
        F: function: dict -> ee.Image, the process model
        Q: function: dict -> ee.Image, the process noise
        H: function: dict -> ee.Image, the measurement function
        R: function: dict -> ee.Image, the measurement noise
        num_params: int, number of parameters in the state, used to create an
            identity matrix with the proper size.

    Returns:
        ee.ImageCollection, the result of applying a Kalman Filter to the input
        collection. Each image in the collection will have one band containing
        the measurement (named z), one band containing the predicted measurement
        (named zprime) and two bands per parameter in the state: one containing
        the state variable's value (named a, b, c, ...) and one containing the
        state variable's covariance (name a_cov, b_cov, c_cov, ...).
    """
    I = ee.Image(ee.Array.identity(num_params))

    def _iterator(curr, prev):
        """Kalman Filter Loop."""
        curr = ee.Image(curr).unmask(UNMASK_VALUE, sameFootprint=False)
        prev = ee.List(prev)

        last = ee.Image(prev.get(-1))
        x = last.select(STATE)
        P = last.select(COV)

        z = curr.toArray().toArray(1).rename(MEASUREMENT)
        t = curr.date().difference(START_DATE, "year")

        kwargs = {"x": x, "P": P, "z": z, "t": t}

        x_bar, P_bar = predict(x, P, F(**kwargs), Q(**kwargs))
        x, P = update(x_bar, P_bar, z, H(**kwargs), R(**kwargs), I)

        # TODO: better name for this? Jeff calls it predictedMeasurement.
        # That confused me because the Kalman Filter predicts the state and then
        # compares that against a measurement. This is the result of using the
        # updated state to better estimate the measurement.
        zprime = H(**kwargs).matrixMultiply(x)

        return ee.List(prev).add(
            ee.Image.cat(
                z.rename(MEASUREMENT),
                x.rename(STATE),
                P.rename(COV),
                zprime.rename(ZPRIME),
            )
        )

    init = ee.Image(init_x).rename(STATE).addBands(ee.Image(init_P).rename(COV))

    # slice off the initial image
    result = ee.ImageCollection(ee.List(collection.iterate(_iterator, [init])).slice(1))

    # use first 'num_params' letters in the alphabet as parameter names
    parameter_names = list(string.ascii_lowercase)[:num_params]

    def _dearrayify(image):
        """Convert array images into a more interpretable form."""
        z = image.select(MEASUREMENT).arrayGet((0, 0))
        zprime = image.select(ZPRIME).arrayGet((0, 0))
        x = image.select(STATE).arrayProject([0]).arrayFlatten([parameter_names])
        P = (
            image.select(COV)
            .matrixDiagonal()
            .arrayProject([0])
            .arrayFlatten([[param + "_cov" for param in parameter_names]])
        )
        return ee.Image.cat(z, zprime, x, P)

    return result.map(_dearrayify)


def F_fn(**kwargs):
    return ee.Image(ee.Array.identity(3))


def Q_fn(**kwargs):
    return ee.Image(ee.Array([[0.001, 0.0005, 0.00025]]).transpose().matrixToDig())


def H_fn(t, **kwargs):
    t = t.multiply(2 * math.pi)
    H = ee.Image.cat(ee.Image.constant(1.0), t.cos(), t.sin()).toArray(0)
    return H.arrayReshape(ee.Image(ee.Array([1, -1])), 2)


def R_fn(**kwargs):
    return ee.Image(ee.Array([[0.1234]]))
