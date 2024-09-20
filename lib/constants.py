""" Constants used throughout project. """

# band names
STATE = "x"
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
    "probabilities": False,
    "bulc_layers": True,
    "confidence": False,
    "final_class": True,
    "final_probabilities": False,
}
