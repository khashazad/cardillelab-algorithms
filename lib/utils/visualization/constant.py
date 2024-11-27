import enum

FIXED_Y_AXIS_LIMIT = 0.4
ASPECT_RATIO = (10, 6)


class PlotType(enum.Enum):
    KALMAN_FIT = "Kalman Fit"
    KALMAN_VS_HARMONIC = "Kalman vs Harmonic Trend"
    KALMAN_RETROFITTED = "Kalman Retrofitted"
    KALMAN_COEFS = "Kalman Coefficients"
    KALMAN_VS_CCDC = "Kalman vs CCDC"
    KALMAN_VS_CCDC_COEFS = "Kalman vs CCDC - Coefficients"
    RESIDUALS_OVER_TIME = "Residuals Over Time"
    KALMAN_AMPLITUDE = "Amplitude"
    BULC_PROBS = "Bulc Probs"


PLOT_TYPES = [
    PlotType.KALMAN_FIT,
    PlotType.KALMAN_VS_HARMONIC,
    PlotType.KALMAN_RETROFITTED,
    PlotType.KALMAN_VS_CCDC,
    PlotType.KALMAN_VS_CCDC_COEFS,
    PlotType.RESIDUALS_OVER_TIME,
    PlotType.KALMAN_AMPLITUDE,
    PlotType.BULC_PROBS,
]

FRAC_OF_YEAR = "fractional_year"
