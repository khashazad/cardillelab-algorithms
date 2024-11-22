""" Simple version of BULC to track process/measurement noise inside an EKF """

import ee
import numpy as np
from scipy.integrate import quad
from pprint import pprint

from lib import constants
from utils.ee.array_operations import dampen_truth_and_add_dummy_row11
from utils import utils
from lib.constants import *

MIN_Z_SCORE = 0
MAX_Z_SCORE = 5

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


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


def update(x_bar, P_bar, z, H, R, num_params):
    """Performs the update step of the Kalman Filter loop.

    Args:
        x_bar: ee array image (n x 1), the predicted state
        P_bar: ee array image (n x n), the predicted state covariance
        z: ee array Image (1 x 1), the measurement
        H: ee array image (1 x n), the measurement function
        R: ee array Image (1 x 1), the measurement noise
        num_params: int, the number of parameters in the state variable

    Returns:
        x (ee array image), P (ee array image): the updated state and state
        covariance
    """
    identity = ee.Image(ee.Array.identity(num_params))

    y = z.subtract(H.matrixMultiply(x_bar))
    S = H.matrixMultiply(P_bar).matrixMultiply(H.matrixTranspose()).add(R)
    S_inv = S.matrixInverse()
    K = P_bar.matrixMultiply(H.matrixTranspose()).matrixMultiply(S_inv)
    x = x_bar.add(K.matrixMultiply(y))
    P = (identity.subtract(K.matrixMultiply(H))).matrixMultiply(P_bar)
    return x, P


def create_priors_from_landcover_map(
    initializing_leveler,
    initial_image,
    number_of_classes_to_track,
):
    def level_initial_lc_exploded(
        lc_map_exploded, initializing_leveler, number_of_classes_to_track
    ):
        leveling_minimum = (
            initializing_leveler.multiply(-1).add(1).divide(number_of_classes_to_track)
        )

        # Final calculation: lc_map_exploded * this_run_initializing_leveler + leveling_minimum
        return lc_map_exploded.multiply(initializing_leveler).add(leveling_minimum)

    # Unmask and multiply the initial image (Earth Engine-specific operations)
    initial_image = ee.Image(initial_image).unmask().multiply(1)

    # Create a list of numbers from 1 to number_of_classes_to_track and convert to an array
    array_list = ee.Array(ee.List.sequence(1, ee.Number(number_of_classes_to_track)))

    # Create an array image from the array of values
    stack_as_array = ee.Image(array_list).rename(PROBABILITY_SELECTOR)

    # Explode the land cover map, setting ith slot to 1 and others to 0
    lc_map_exploded = stack_as_array.eq(initial_image)

    # Level the exploded map (placeholder for actual function)
    lc_map_exploded_and_leveled = level_initial_lc_exploded(
        lc_map_exploded, initializing_leveler, number_of_classes_to_track
    )

    return lc_map_exploded_and_leveled


def overwrite_named_slot(mb_stack, layer_values, layer_name):
    return ee.Image(mb_stack).addBands(
        layer_values, [layer_name], True
    )  # It overwrites because the third parameter is True


def extract_named_slot(mb_stack, layer_name):
    return ee.Image(mb_stack).select(layer_name)


def day_i_rebalancing_v3(
    prob_stack_to_level, balance_factor, minimum_prob_to_add_daily
):
    prob_stack_to_level = ee.Image(prob_stack_to_level)
    return prob_stack_to_level.multiply(balance_factor).add(minimum_prob_to_add_daily)


def increment_global_image_counter(accumulating_answer):
    one_value = ee.Number(accumulating_answer.get(GLOBAL_COUNTER)).add(1)
    return accumulating_answer.set(GLOBAL_COUNTER, one_value)


def create_1d_transition_array_image_from_dynamic_truth(
    one_event, any_dampened_truth_table, list_of_event_classes_from_0
):
    # Convert list_of_event_classes_from_0 to an Earth Engine list
    list_of_event_classes_from_0 = ee.List(list_of_event_classes_from_0)

    def afn_sliced_2d_to_remappable(one_sliced_truth_table_2d):
        # Cast the table to an array, then to a list, and map over the list to return each row as an array
        the_2d_as_list = ee.Array(one_sliced_truth_table_2d).toList()
        the_2d_transition = the_2d_as_list.map(lambda l: ee.Array(l))
        return the_2d_transition

    # Convert the dampened truth table to a list of arrays
    dampened_truth_table_as_list_of_arrays = afn_sliced_2d_to_remappable(
        any_dampened_truth_table
    )

    # Remap the event values to corresponding truth table rows
    truth_table_row_to_array_image = (
        ee.Image(one_event)
        .remap(list_of_event_classes_from_0, dampened_truth_table_as_list_of_arrays)
        .rename([CONDITIONAL_STACK])
    )

    return truth_table_row_to_array_image


def build_start_probs(bbiidd):
    # Unpack the parameter dictionary
    initialization_approach = bbiidd.get("initialization_approach")
    initializing_leveler = bbiidd.get("initializing_leveler")
    base_land_cover_image = bbiidd.get("base_land_cover_image")
    number_of_classes_to_track = bbiidd.get("number_of_classes_to_track")
    class_name_list = bbiidd.get("class_name_list")

    # Handle different initialization approaches
    if initialization_approach == "F":  # First run image
        # Create priors from the land cover map (placeholder for actual Earth Engine function)
        current_probs_array_img = create_priors_from_landcover_map(
            initializing_leveler,
            base_land_cover_image,
            number_of_classes_to_track,
        )
        current_probs = current_probs_array_img.rename(PROBABILITY_SELECTOR)

    return current_probs


def build_comparison_layer(cciidd):
    # Unpack the parameter dictionary
    transition_creation_method = cciidd.get("transition_creation_method")
    comparison_layer_label = cciidd.get("comparison_layer_label")
    initialization_approach = cciidd.get("initialization_approach")
    first_comparison_image = cciidd.get("first_comparison_image")

    # Develop the comparison layer if needed
    if (
        transition_creation_method == "C"
    ):  # Custom-passed matrix, no comparison layer needed
        comparison_layer = ee.Image(-1).rename(comparison_layer_label)

    elif transition_creation_method == "V":  # Overlay approach
        if initialization_approach == "F":  # First run image
            comparison_layer = ee.Image(first_comparison_image).rename(
                [comparison_layer_label]
            )

    return comparison_layer


def initialize_iterate_package(iidd):
    initializing_leveler = iidd.get("initializing_leveler")
    transition_creation_method = iidd.get("transition_creation_method")
    number_of_classes_to_track = iidd.get("number_of_classes_to_track")
    default_study_area = iidd.get("default_study_area")
    class_name_list = iidd.get("class_name_list")
    initialization_approach = iidd.get("initialization_approach")
    base_land_cover_image = iidd.get("base_land_cover_image")
    first_comparison_image = iidd.get("first_comparison_image")
    kalman_init_image = iidd.get("kalman_init_image")
    # Build the start probabilities parameter dictionary
    bbiidd = {
        "initialization_approach": initialization_approach,
        "initializing_leveler": initializing_leveler,
        "base_land_cover_image": base_land_cover_image,
        "number_of_classes_to_track": number_of_classes_to_track,
        "default_study_area": default_study_area,
        "class_name_list": class_name_list,
    }

    # Build initial probabilities
    current_probs = build_start_probs(bbiidd)

    # Initialize accumulating_answer with current probabilities
    accumulating_answer = current_probs

    # Build the comparison layer parameter dictionary
    cciidd = {
        "transition_creation_method": transition_creation_method,
        "comparison_layer_label": COMPARISON_LAYER,
        "initialization_approach": initialization_approach,
        "base_land_cover_image": base_land_cover_image,
        "first_comparison_image": first_comparison_image,
        "default_study_area": default_study_area,
    }

    # Build the first comparison layer
    the_comparison_layer = build_comparison_layer(cciidd)
    accumulating_answer = accumulating_answer.addBands(the_comparison_layer)

    # Add image counters
    the_total_image_counter = ee.Image(0).rename([GLOBAL_COUNTER])
    the_per_pixel_image_counter = ee.Image(0).rename([PER_PIXEL_IMAGE_COUNTER])
    accumulating_answer = accumulating_answer.addBands(
        the_total_image_counter, [GLOBAL_COUNTER], False
    )
    accumulating_answer = accumulating_answer.addBands(
        the_per_pixel_image_counter, [PER_PIXEL_IMAGE_COUNTER], False
    )

    # Add the iteration counter as a property
    accumulating_answer = accumulating_answer.set(GLOBAL_COUNTER_PROPERTY_NAME, 0)

    # Handle probabilities initialization based on the approach
    if initialization_approach == "f":  # First run
        if RECORDING_FLAGS["probabilities"]:
            current_probs_multi_band = current_probs.arrayFlatten([class_name_list])
            accumulating_answer = accumulating_answer.addBands(current_probs_multi_band)

    # Handle BULC layer and confidence if required
    if RECORDING_FLAGS["bulc_layers"]:
        bulc_posterior = (
            current_probs.arrayArgmax().arrayFlatten([[BULC_CLASSIFICATION]]).add(1)
        )
        accumulating_answer = accumulating_answer.addBands(
            bulc_posterior, [BULC_CLASSIFICATION], False
        )

    if RECORDING_FLAGS["confidence"]:
        bulc_confidence_1d_array = current_probs.arrayReduce(
            ee.Reducer.max(), [0]
        ).select(0)
        bulc_confidence_0d = bulc_confidence_1d_array.reduce(
            ee.Reducer.max()
        ).arrayFlatten([[BULC_CONFIDENCE]])
        accumulating_answer = accumulating_answer.addBands(bulc_confidence_0d)

    accumulating_answer = accumulating_answer.addBands(
        kalman_init_image, [constants.STATE, constants.COV]
    )

    # Return the final accumulating answer
    return accumulating_answer


def hidden_bulc_iterate_with_options(
    one_event,
    accumulating_answer,
    truth_table_stack,
    transition_creation_method,
    list_of_event_classes_from_0,
    posterior_leveler,
    posterior_minimum,
    number_of_classes_to_track,
    class_name_list,
    transition_leveler,
    Q,
    R,
    F,
    H,
    num_params,
    measurement_band,
    Q_change_threshold,
    Q_scale_factor,
):

    accumulating_answer = ee.Image(accumulating_answer)

    curr = ee.Image(one_event.select(measurement_band))

    one_event = ee.Image(one_event.select("Slot"))

    # Assume all values are valid
    one_event_valid_values = one_event

    # Extract prior probabilities and comparison layer from accumulating answer
    current_probs = accumulating_answer.select(PROBABILITY_SELECTOR)
    comparison_layer = accumulating_answer.select(COMPARISON_LAYER)

    # Update image counters
    total_image_counter = (
        accumulating_answer.select(GLOBAL_COUNTER).add(1).rename([GLOBAL_COUNTER])
    )

    accumulating_answer = overwrite_named_slot(
        accumulating_answer, total_image_counter, GLOBAL_COUNTER
    )

    if RECORDING_FLAGS["iteration_number"]:
        accumulating_answer = accumulating_answer.addBands(
            total_image_counter.rename(["iter"])
        )

    per_pixel_image_counter = accumulating_answer.select(PER_PIXEL_IMAGE_COUNTER)

    per_pixel_image_counter = per_pixel_image_counter.where(
        one_event_valid_values, per_pixel_image_counter.add(1)
    ).rename([PER_PIXEL_IMAGE_COUNTER])

    accumulating_answer = overwrite_named_slot(
        accumulating_answer, per_pixel_image_counter, PER_PIXEL_IMAGE_COUNTER
    )

    if RECORDING_FLAGS["events"]:
        accumulating_answer = accumulating_answer.addBands(
            one_event.mask(one_event_valid_values).rename([EVENT])
        )

    # Handle different transition creation methods
    if transition_creation_method == "C":
        transition_array_image_rescaled_and_dampened = (
            create_1d_transition_array_image_from_dynamic_truth(
                one_event,
                truth_table_stack,
                list_of_event_classes_from_0,
            )
        )

    # Record conditionals if needed
    if RECORDING_FLAGS["conditionals"]:
        accumulating_answer = accumulating_answer.addBands(
            transition_array_image_rescaled_and_dampened
        )

    # Apply Bayes to update probabilities
    posterior_probs_valid_pixels_1d = (
        current_probs.mask(one_event_valid_values)
        .multiply(transition_array_image_rescaled_and_dampened)
        .divide(
            current_probs.arrayDotProduct(transition_array_image_rescaled_and_dampened)
        )
    )

    # Rebalance and dampen probabilities
    posterior_probs_valid_pixels_1d = ee.Image(
        day_i_rebalancing_v3(
            posterior_probs_valid_pixels_1d, posterior_leveler, posterior_minimum
        )
    )

    # Burn pixel updates onto prior probabilities
    posterior_probs_all_pixels_1d = current_probs.where(
        one_event_valid_values, posterior_probs_valid_pixels_1d
    ).rename([PROBABILITY_SELECTOR])
    accumulating_answer = accumulating_answer.addBands(
        posterior_probs_all_pixels_1d, [PROBABILITY_SELECTOR], True
    )

    if RECORDING_FLAGS["probabilities"]:
        accumulating_answer = accumulating_answer.addBands(
            posterior_probs_all_pixels_1d.arrayFlatten([class_name_list])
        )

    # Record BULC layers and confidence if needed
    if RECORDING_FLAGS["bulc_layers"]:
        bulc_posterior = (
            posterior_probs_all_pixels_1d.arrayArgmax()
            .arrayFlatten([[BULC_CLASSIFICATION]])
            .add(1)
        )
        accumulating_answer = accumulating_answer.addBands(
            bulc_posterior, [BULC_CLASSIFICATION], False
        )

    if RECORDING_FLAGS["confidence"]:
        bulc_confidence_1d_array = posterior_probs_all_pixels_1d.arrayReduce(
            ee.Reducer.max(), [0]
        ).select(0)
        bulc_confidence_0d = bulc_confidence_1d_array.reduce(
            ee.Reducer.max()
        ).arrayFlatten([[BULC_CONFIDENCE]])
        accumulating_answer = accumulating_answer.addBands(bulc_confidence_0d)

    UNMASK_VALUE = -999

    x_prev = accumulating_answer.select(
        ee.String(accumulating_answer.select(f".*{STATE}.*").bandNames().get(-1))
    )
    P_prev = accumulating_answer.select(
        ee.String(accumulating_answer.select(f".*{COV}.*").bandNames().get(-1))
    )

    probability_of_no_change = ee.Image(
        posterior_probs_all_pixels_1d.arrayFlatten([class_name_list]).select(
            [class_name_list.get(1)]
        )
    )

    z = curr.select(measurement_band).toArray().toArray(1)
    t = curr.date().difference("2016-01-01", "year")

    adaptive_Q = ee.Image(Q).where(
        probability_of_no_change.lt(Q_change_threshold),
        ee.Image(Q).multiply(ee.Number(Q_scale_factor)),
    )

    x_bar, P_bar = predict(x_prev, P_prev, F(**locals()), adaptive_Q)
    x, P = update(x_bar, P_bar, z, H(**locals()), R, num_params)

    x = x_prev.where(curr.select(measurement_band).gt(UNMASK_VALUE), x)
    P = P_prev.where(curr.select(measurement_band).gt(UNMASK_VALUE), P)

    pixel_image = x.arrayProject([0]).arrayFlatten([["INTP", "COS0", "SIN0"]])

    intp = pixel_image.select("INTP")
    cos = pixel_image.select("COS0")
    sin = pixel_image.select("SIN0")

    phi = t.multiply(ee.Number(6.283))
    estimate = intp.add(cos.multiply(phi.cos())).add(sin.multiply(phi.sin()))

    result = [
        curr.rename(constants.MEASUREMENT_LABEL),
        x.rename(constants.STATE),
        P.rename(constants.COV),
        estimate.rename(constants.ESTIMATE_LABEL),
        ee.Image(curr.date().millis()).rename(constants.TIMESTAMP_LABEL),
    ]

    accumulating_answer = accumulating_answer.addBands(ee.Image.cat(*result))

    # Increment the global image counter
    accumulating_answer = ee.Image(increment_global_image_counter(accumulating_answer))

    return accumulating_answer


def kalman_with_bulcd(args):
    bulc_args = args["kalman_with_bulcd_params"]["bulc_arguments"]
    args = args["kalman_with_bulcd_params"]

    # Parse the parameter dictionary
    kalman_params = args["kalman_params"]
    Q = kalman_params["Q"]
    R = kalman_params["R"]
    F = kalman_params["F"]
    H = kalman_params["H"]
    num_params = kalman_params["num_params"]
    measurement_band = kalman_params["measurement_band"]
    Q_change_threshold = kalman_params["Q_change_threshold"]
    Q_scale_factor = kalman_params["Q_scale_factor"]

    events_as_image_collection = ee.ImageCollection(args.get("events_and_measurements"))
    initializing_leveler = bulc_args.getNumber("initializing_leveler")
    posterior_leveler = bulc_args.getNumber("posterior_leveler")
    posterior_minimum = bulc_args.getNumber("posterior_minimum")
    default_study_area = ee.Geometry(bulc_args.get("default_study_area"))
    max_class_in_event_images = bulc_args.getNumber("max_class_in_event_images")
    number_of_classes_to_track = bulc_args.getNumber("number_of_classes_to_track")

    initialization_approach = args.get("initialization_approach")

    if initialization_approach == "F":
        base_land_cover_image = args.get("base_land_cover_image")

    transition_creation_method = args.get("transition_creation_method")
    if transition_creation_method == "C":
        custom_transition_matrix = bulc_args.get("custom_transition_matrix")
        transition_leveler = bulc_args.getNumber("transition_leveler")
        transition_minimum = bulc_args.getNumber("transition_minimum")

    list_of_event_classes_from_0 = ee.List.sequence(0, max_class_in_event_images)
    list_of_tracked_classes_from_1 = ee.List.sequence(1, number_of_classes_to_track)

    class_name_list = list_of_tracked_classes_from_1.map(
        lambda x: ee.String(PROBABILITY_SELECTOR).cat(ee.String(ee.Number(x).int()))
    )

    # Create an initialization dictionary for the iterate package
    iidd = {
        "initializing_leveler": initializing_leveler,
        "transition_creation_method": transition_creation_method,
        "number_of_classes_to_track": number_of_classes_to_track,
        "default_study_area": default_study_area,
        "class_name_list": class_name_list,
        "initialization_approach": initialization_approach,
        "base_land_cover_image": (
            base_land_cover_image if initialization_approach == "F" else None
        ),
        "first_comparison_image": bulc_args.get("first_comparison_image"),
        "kalman_init_image": kalman_params["init_image"],
    }

    accumulating_answer = initialize_iterate_package(iidd)

    if transition_creation_method == "C":
        # Convert the custom transition matrix to an Earth Engine array
        truth_table_stack = ee.Array(custom_transition_matrix)

        # Dampen the truth table and add a dummy row
        truth_table_stack = dampen_truth_and_add_dummy_row11(
            custom_transition_matrix, transition_leveler, transition_minimum
        )

    accumulating_answer = accumulating_answer.set(
        "truth_table_stack", truth_table_stack
    )

    the_bound_iterate = ee.Image(
        events_as_image_collection.iterate(
            lambda image, state: hidden_bulc_iterate_with_options(
                image,
                state,
                truth_table_stack,
                transition_creation_method,
                list_of_event_classes_from_0,
                posterior_leveler,
                posterior_minimum,
                number_of_classes_to_track,
                class_name_list,
                transition_leveler,
                Q,
                R,
                F,
                H,
                num_params,
                measurement_band,
                Q_change_threshold,
                Q_scale_factor,
            ),
            accumulating_answer,
        )
    )

    if RECORDING_FLAGS["confidence"]:
        # Get the Probability layers and put them in their own image
        layer_match_key = ".*" + BULC_CONFIDENCE + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_confidence_layers = all_matched_layers_as_multiband

    if RECORDING_FLAGS["events"]:
        # Get the Event layers and put them in their own image
        layer_match_key = ".*" + EVENT + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_event_layers = all_matched_layers_as_multiband

    if RECORDING_FLAGS["bulc_layers"]:
        # Get the BULC layers and put them in their own image
        layer_match_key = ".*" + BULC_CLASSIFICATION + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_bulc_layers = all_matched_layers_as_multiband

    if RECORDING_FLAGS["probabilities"]:
        # Get the Probability layers and put them in their own image
        layer_match_key = ".*" + PROBABILITY_SELECTOR + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_probability_layers = all_matched_layers_as_multiband

    if RECORDING_FLAGS["final_class"]:
        # Grab the final BULC probabilities
        final_bulc_probs = (
            the_bound_iterate.select(0)
            .arrayFlatten([class_name_list])
            .rename(class_name_list)
        )

    all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
    all_bulc_layers = all_matched_layers_as_multiband

    return {
        "multi_band_bulc_return": the_bound_iterate.clip(default_study_area),
        # "all_confidence_layers": all_confidence_layers.clip(default_study_area),
        # "all_event_layers": all_event_layers.selfMask().clip(default_study_area),
        "all_bulc_layers": all_bulc_layers.selfMask().clip(default_study_area),
        "all_probability_layers": all_probability_layers.clip(default_study_area),
        "kalman_states": the_bound_iterate.select(".*state.*").clip(default_study_area),
        "kalman_covariances": the_bound_iterate.select(".*P.*").clip(
            default_study_area
        ),
        "kalman_estimates": the_bound_iterate.select(".*estimate.*").clip(
            default_study_area
        ),
        "kalman_dates": the_bound_iterate.select(".*date.*").clip(default_study_area),
        "kalman_measurements": the_bound_iterate.select(".*z.*").clip(
            default_study_area
        ),
        "final_bulc_probs": final_bulc_probs.clip(default_study_area),
        "default_study_area": default_study_area,
    }
