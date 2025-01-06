import enum

FIXED_Y_AXIS_LIMIT = 0.4
ASPECT_RATIO = (10, 6)


class PlotType(enum.Enum):
    KALMAN_FIT = "Kalman Fit"
    KALMAN_VS_HARMONIC = "Kalman vs Harmonic Trend"
    KALMAN_RETROFITTED = "Kalman Retrofitted"
    KALMAN_YEARLY_FIT = "Kalman Yearly Fit"
    KALMAN_COEFS = "Kalman Coefficients"
    KALMAN_VS_CCDC = "Kalman vs CCDC"
    KALMAN_VS_CCDC_COEFS = "Kalman vs CCDC - Coefficients"
    RESIDUALS = "Residuals"
    KALMAN_AMPLITUDE = "Amplitude"
    BULC_PROBS = "Bulc Probs"


PLOT_TYPES = [
    PlotType.KALMAN_FIT,
    PlotType.KALMAN_VS_HARMONIC,
    PlotType.KALMAN_YEARLY_FIT,
    PlotType.KALMAN_RETROFITTED,
    PlotType.KALMAN_VS_CCDC,
    PlotType.KALMAN_VS_CCDC_COEFS,
    PlotType.RESIDUALS,
    PlotType.KALMAN_AMPLITUDE,
    PlotType.BULC_PROBS,
]

COLOR_PALETTE_10 = [
    "#f2d5cf",  # Rosewater
    "#eebebe",  # Flamingo
    "#f4b8e4",  # Pink
    "#ca9ee6",  # Mauve
    "#e78284",  # Red
    "#ea999c",  # Maroon
    "#ef9f76",  # Peach
    "#e5c890",  # Yellow
    "#a6d189",  # Green
    "#81c8be",  # Teal
]

FRAC_OF_YEAR = "fractional_year"
