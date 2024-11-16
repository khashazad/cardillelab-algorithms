""" Constants used throughout project. """

import enum

RESIDUAL = "residual"
CHANGE_PROB = "change_prob"
ESTIMATE = "estimate"
TIMESTAMP = "timestamp"
FRACTION_OF_YEAR = "frac_of_year"
DATE = "date"
AMPLITUDE = "amplitude"
MEASUREMENT = "measurement"

PROBABILITY_LABEL = "probability_array"
PROBABILITY_SELECTOR = "probability_class"
PER_PIXEL_IMAGE_COUNTER = "per_pixel_image_counter"
BULC_CONFIDENCE = "BULC_confidence"
BULC_CLASSIFICATION = "BULC_classification"
KALMAN_STATE = "kalman_state"
KALMAN_STATE_COV = "kalman_state_covariance"
GLOBAL_COUNTER = "number_of_images_so_far"
CONDITIONAL_STACK = "conditional_stack"
COMPARISON_LAYER = "comparison_layer"
EVENT = "event"
GLOBAL_COUNTER_PROPERTY_NAME = "number_of_images_so_far"

RECORDING_FLAGS = {
    "iteration_number": False,
    "events": False,
    "conditionals": False,
    "probabilities": True,
    "bulc_layers": True,
    "confidence": False,
    "final_class": True,
    "final_probabilities": True,
    "ccdc_coefficients": True,
    "estimate": True,
    "measurement": True,
    "timestamp": True,
}


class KalmanRecordingFlags(enum.Enum):
    STATE = "state"
    STATE_COV = "state_covariance"
    ESTIMATE = ESTIMATE
    TIMESTAMP = TIMESTAMP
    FRACTION_OF_YEAR = FRACTION_OF_YEAR
    AMPLITUDE = AMPLITUDE
    MEASUREMENT = MEASUREMENT
    CCDC_COEFFICIENTS = "ccdc_coefficients"


class BulcRecordingFlags(enum.Enum):
    ITERATION_NUMBER = "iteration_number"
    EVENTS = "events"
    CONDITIONALS = "conditionals"
    PROBABILITIES = "probabilities"
    BULC_LAYERS = "bulc_layers"
    CONFIDENCE = "confidence"
    FINAL_CLASS = "final_class"
    FINAL_PROBABILITIES = "final_probabilities"


class Index(enum.Enum):
    SWIR = "swir"
    NBR = "nbr"
    NDVI = "ndvi"


class Sensor(enum.Enum):
    L7 = "L7"
    L8 = "L8"
    L9 = "L9"
    S2 = "S2"


class Kalman(enum.Enum):
    F = "F"  # state transition model
    Q = "Q"  # process noise covariance
    H = "H"  # observation model
    R = "R"  # measurement noise covariance
    P = "P"  # state covariance matrix
    X = "X"  # state
    Z = "z"  # observation
    INITIAL_STATE = "initial_state"
    COV_PREFIX = "cov"


class Initialization(enum.Enum):
    UNIFORM = "uniform"
    POSTHOC = "posthoc"


class Harmonic(enum.Enum):
    INTERCEPT = "intercept"
    SLOPE = "slope"
    COS = "cos"
    SIN = "sin"
    COS2 = "cos2"
    SIN2 = "sin2"
    COS3 = "cos3"
    SIN3 = "sin3"
    UNIMODAL = "unimodal"
    BIMODAL = "bimodal"
    TRIMODAL = "trimodal"
    FIT = "fit"


NUM_MEASURES = 1  # eeek only supports one band at a time

MASK_VALUE = -999

POINT_INDEX = "point_index"
