""" Simple version of BULC to track process/measurement noise inside an EKF """
import ee
import numpy as np
from scipy.integrate import quad

from eeek import constants

MIN_Z_SCORE = 0
MAX_Z_SCORE = 5


def bulcp_update(curr, last, leveler=0.1, num_classes=3):
    """Run one update step of BULC-P

    Caller is responsible to store the result.

    Args:
        curr: ee.Image
        last: ee.Image

    Returns:
        ee.Image
    """
    curr = ee.Image(curr).toArray()
    last = ee.Image(last).toArray()

    min_prob = ee.Number(1).subtract(leveler).divide(num_classes)

    update = last.multiply(curr).divide(last.arrayDotProduct(curr))
    dampened = update.multiply(leveler).add(min_prob)

    output = last.where(dampened.mask(), dampened)
    return output


def build_z_table(min_z_score, max_z_score):
    """Constructs an ee array image z-table.

    Only computes values for z-scores up to two decimal places.

    Args:
        min_z_score: lowest z-score to calculate in table.
        max_z_score: highest z-score to calculate in table.

    Returns:
        ee.Image
    """

    def pdf(x):
        """Standard normal probability distribution function."""
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp((-(x**2)) / 2.0)

    def cdf(x):
        """Standard normal cummulative distribution function."""
        val, _ = quad(pdf, np.NINF, x)
        return val

    num = 100 * (max_z_score - min_z_score)
    z_scores = np.around(np.linspace(min_z_score, max_z_score, num), 2)
    return ee.Image(ee.Array([cdf(x) for x in z_scores]))


Z_TABLE = build_z_table(MIN_Z_SCORE, MAX_Z_SCORE)


def get_change_prob(im):
    """Given an image of z-scores return an image of change probabilities.

    Uses a precomputed z-table to approximate change probabilities.

    Args:
        im: ee.Image with one band containing the z-score for each pixel.

    Returns:
        ee.Image with three bands containing the probability that the band has
        increased, remained stable, or decreased.
    """
    index_im = im.abs().multiply(100).round().clamp(MIN_Z_SCORE, MAX_Z_SCORE)
    prob = Z_TABLE.arrayGet(index_im.toInt())
    unchanged_prob = ((prob.multiply(-1)).add(1)).multiply(2)
    changed_prob = (unchanged_prob.multiply(-1)).add(1)

    increased_prob = changed_prob.multiply(im.gte(0))
    decreased_prob = changed_prob.multiply(im.lt(0))
    return ee.Image.cat(increased_prob, unchanged_prob, decreased_prob)


def preprocess(bulc_leveler=0.1):
    """Calcualtes the probability of change using z-scores and BULC-P.

    Can be used to create a preprocess_fn

    Args:
        bulc_leveler: float

    Returns:
        function float -> list[ee.Image]
    """

    def inner(curr, prev, z, H, x, **kwargs):
        curr = ee.Image(curr)
        prev = ee.List(prev)

        curr_residual = z.subtract(H(**kwargs).matrixMultiply(x)).arrayGet((0, 0))
        prev_residuals = ee.ImageCollection(prev).select(constants.RESIDUAL)
        mean_residuals = prev_residuals.reduce(ee.Reducer.mean())
        std_residuals = prev_residuals.reduce(ee.Reducer.stdDev())

        z_score = curr_residual.subtract(mean_residuals).divide(std_residuals)
        change_prob = get_change_prob(z_score)
        smoothed_change_prob = bulcp_update(
            change_prob,
            ee.Image(prev.get(-1)).select(constants.CHANGE_PROB),
            bulc_leveler,
        )

        return [
            curr_residual.rename(constants.RESIDUAL),
            smoothed_change_prob.rename(constants.CHANGE_PROB),
        ]

    return inner


def bulc_as_noise(preprocess_results, num_params, **kwargs):
    """Converts the output of preprocess into a noise matrix.

    Can be used as Q.

    Args:
        preprocess_results: list[ee.Image], 2nd element should be the bulc
            probabilities.
        num_params: int

    Returns:
        ee.Image
    """
    smoothed_change_prob = preprocess_results[1]

    return (
        smoothed_change_prob.arrayGet([1])
        .multiply(-1)
        .add(1)
        .arrayRepeat(0, num_params)
        .arrayReshape(ee.Image(ee.Array([1, num_params])), 2)
        .matrixToDiag()
    )
