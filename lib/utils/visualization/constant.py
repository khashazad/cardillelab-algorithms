import enum

FIXED_Y_AXIS_LIMIT = 0.4
ASPECT_RATIO = (12, 8)


class PlotType(enum.Enum):
    KALMAN_FIT = "Kalman Fit"
    KALMAN_VS_HARMONIC = "Kalman vs Harmonic Trend"
    KALMAN_COEFS = "Kalman Coefficients"
    KALMAN_COEFS_VS_CCDC = "Kalman vs CCDC - Coefficients"
    RESIDUALS_OVER_TIME = "Residuals Over Time"
    KALMAN_AMPLITUDE = "Amplitude"
    BULC_PROBS = "Bulc Probs"


PLOT_TYPES = [
    PlotType.KALMAN_FIT,
    PlotType.KALMAN_VS_HARMONIC,
    PlotType.RESIDUALS_OVER_TIME,
    PlotType.KALMAN_AMPLITUDE,
    PlotType.BULC_PROBS,
    PlotType.KALMAN_COEFS_VS_CCDC,
]

FRAC_OF_YEAR = "fractional_year"
