""" Constants used throughout project. """

import enum

# band names
STATE = "state"
COV = "P"
MEASUREMENT = "z"
RESIDUAL = "residual"
CHANGE_PROB = "change_prob"
ESTIMATE = "estimate"
DATE = "date"
AMPLITUDE = "amplitude"
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
}


class Index(enum.Enum):
    SWIR = "swir"
    NBR = "nbr"
    NDVI = "ndvi"


class Sensor(enum.Enum):
    L7 = "L7"
    L8 = "L8"
    L9 = "L9"
    S2 = "S2"
