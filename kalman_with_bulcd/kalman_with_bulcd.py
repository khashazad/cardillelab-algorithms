""" Simple version of BULC to track process/measurement noise inside an EKF """

import ee
import numpy as np
from scipy.integrate import quad
from pprint import pprint

from lib import constants
from utils.ee.array_operations import dampen_truth_and_add_dummy_row11
from utils import utils

MIN_Z_SCORE = 0
MAX_Z_SCORE = 5

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


def create_priors_from_landcover_map(
    this_run_initializing_leveler,
    initial_image,
    number_of_classes_to_track,
    probs_label,
):
    def level_initial_lc_exploded(
        lc_map_exploded, this_run_initializing_leveler, number_of_classes_to_track
    ):
        # Calculate (1 - this_run_initializing_leveler) / number_of_classes_to_track
        leveling_minimum_numerator_part1 = this_run_initializing_leveler.multiply(-1)
        leveling_minimum_numerator = leveling_minimum_numerator_part1.add(1)
        leveling_minimum = leveling_minimum_numerator.divide(number_of_classes_to_track)

        # Final calculation: lc_map_exploded * this_run_initializing_leveler + leveling_minimum
        return lc_map_exploded.multiply(this_run_initializing_leveler).add(
            leveling_minimum
        )

    # Unmask and multiply the initial image (Earth Engine-specific operations)
    initial_image = ee.Image(initial_image).unmask().multiply(1)

    # Create a list of numbers from 1 to number_of_classes_to_track and convert to an array
    array_list = ee.Array(ee.List.sequence(1, ee.Number(number_of_classes_to_track)))

    # Create an array image from the array of values
    stack_as_array = ee.Image(array_list).rename(probs_label)

    # Explode the land cover map, setting ith slot to 1 and others to 0
    lc_map_exploded = stack_as_array.eq(initial_image)

    # Level the exploded map (placeholder for actual function)
    lc_map_exploded_and_leveled = level_initial_lc_exploded(
        lc_map_exploded, this_run_initializing_leveler, number_of_classes_to_track
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


def increment_global_image_counter(accumulating_answer, global_counter_property_name):
    one_value = ee.Number(accumulating_answer.get(global_counter_property_name)).add(1)
    return accumulating_answer.set(global_counter_property_name, one_value)


def create_1d_transition_array_image_from_dynamic_truth(
    one_event, any_dampened_truth_table, list_of_event_classes_from_0, cond_stack_label
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
        .rename([cond_stack_label])
    )

    return truth_table_row_to_array_image


def build_start_probs(bbiidd):
    # Unpack the parameter dictionary
    initialization_approach = bbiidd.get("initialization_approach")
    initializing_leveler = bbiidd.get("initializing_leveler")
    base_land_cover_image = bbiidd.get("base_land_cover_image")
    number_of_classes_to_track = bbiidd.get("number_of_classes_to_track")
    class_name_list = bbiidd.get("class_name_list")
    probs_label = bbiidd.get("probs_label")

    pprint(initialization_approach)
    # Handle different initialization approaches
    if initialization_approach == "F":  # First run image
        # Create priors from the land cover map (placeholder for actual Earth Engine function)
        current_probs_array_img = create_priors_from_landcover_map(
            initializing_leveler,
            base_land_cover_image,
            number_of_classes_to_track,
            probs_label,
        )
        current_probs = current_probs_array_img.rename(probs_label)

    return current_probs


def build_comparison_layer(cciidd):
    # Unpack the parameter dictionary
    transition_creation_method = cciidd.get("transition_creation_method")
    comparison_layer_label = cciidd.get("comparison_layer_label")
    initialization_approach = cciidd.get("initialization_approach")
    initialization_approach_parameter1 = cciidd.get(
        "initialization_approach_parameter1"
    )
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
    probs_label = iidd.get("probs_label")
    comparison_layer_label = iidd.get("comparison_layer_label")
    total_image_counter_label = iidd.get("total_image_counter_label")
    per_pixel_image_counter_label = iidd.get("per_pixel_image_counter_label")
    cond_stack_label = iidd.get("cond_stack_label")
    record_probabilities_at_each_time_step = iidd.get(
        "record_probabilities_at_each_time_step"
    )
    initialization_approach = iidd.get("initialization_approach")
    pprint(initialization_approach)
    base_land_cover_image = iidd.get("base_land_cover_image")
    first_comparison_image = iidd.get("first_comparison_image")
    global_counter_property_name = iidd.get("global_counter_property_name")
    record_bulc_layers_at_each_time_step = iidd.get(
        "record_bulc_layers_at_each_time_step"
    )
    record_confidence_at_each_time_step = iidd.get(
        "record_confidence_at_each_time_step"
    )
    bulc_confidence_label = iidd.get("bulc_confidence_label")
    bulc_layer_label = iidd.get("bulc_layer_label")

    # Build the start probabilities parameter dictionary
    bbiidd = {
        "initialization_approach": initialization_approach,
        "initializing_leveler": initializing_leveler,
        "base_land_cover_image": base_land_cover_image,
        "number_of_classes_to_track": number_of_classes_to_track,
        "default_study_area": default_study_area,
        "class_name_list": class_name_list,
        "probs_label": probs_label,
    }

    # Build initial probabilities (placeholder for afn_build_start_probs)
    current_probs = build_start_probs(bbiidd)

    # Initialize accumulating_answer with current probabilities
    accumulating_answer = current_probs

    # Build the comparison layer parameter dictionary
    cciidd = {
        "transition_creation_method": transition_creation_method,
        "comparison_layer_label": comparison_layer_label,
        "initialization_approach": initialization_approach,
        "base_land_cover_image": base_land_cover_image,
        "first_comparison_image": first_comparison_image,
        "default_study_area": default_study_area,
    }

    # Build the first comparison layer (placeholder for afn_build_comparison_layer)
    the_comparison_layer = build_comparison_layer(cciidd)
    accumulating_answer = accumulating_answer.addBands(the_comparison_layer)

    # Add image counters
    the_total_image_counter = ee.Image(0).rename([total_image_counter_label])
    the_per_pixel_image_counter = ee.Image(0).rename([per_pixel_image_counter_label])
    accumulating_answer = accumulating_answer.addBands(
        the_total_image_counter, [total_image_counter_label], False
    )
    accumulating_answer = accumulating_answer.addBands(
        the_per_pixel_image_counter, [per_pixel_image_counter_label], False
    )

    # Add the iteration counter as a property
    accumulating_answer = accumulating_answer.set(global_counter_property_name, 0)

    # Handle probabilities initialization based on the approach
    if initialization_approach == "f":  # First run
        if record_probabilities_at_each_time_step:
            current_probs_multi_band = current_probs.arrayFlatten([class_name_list])
            accumulating_answer = accumulating_answer.addBands(current_probs_multi_band)

    # Handle BULC layer and confidence if required
    if record_bulc_layers_at_each_time_step:
        bulc_posterior = (
            current_probs.arrayArgmax().arrayFlatten([[bulc_layer_label]]).add(1)
        )
        accumulating_answer = accumulating_answer.addBands(
            bulc_posterior, [bulc_layer_label], False
        )

    if record_confidence_at_each_time_step:
        bulc_confidence_1d_array = current_probs.arrayReduce(
            ee.Reducer.max(), [0]
        ).select(0)
        bulc_confidence_0d = bulc_confidence_1d_array.reduce(
            ee.Reducer.max()
        ).arrayFlatten([[bulc_confidence_label]])
        accumulating_answer = accumulating_answer.addBands(bulc_confidence_0d)

    # Return the final accumulating answer
    return accumulating_answer


def hidden_bulc_iterate_with_options(
    one_event,
    accumulating_answer,
    probs_label,
    event_label,
    comparison_layer_label,
    total_image_counter_label,
    per_pixel_image_counter_label,
    record_iteration_number_at_each_time_step,
    record_events_at_each_time_step,
    record_conditionals_at_each_time_step,
    truth_table_stack,
    transition_creation_method,
    list_of_event_classes_from_0,
    posterior_leveler,
    posterior_minimum,
    number_of_classes_to_track,
    record_probabilities_at_each_time_step,
    class_name_list,
    transition_leveler,
    global_counter_property_name,
    record_bulc_layers_at_each_time_step,
    record_confidence_at_each_time_step,
    bulc_layer_label,
    bulc_confidence_label,
    cond_stack_label,
    record_final_class,
):

    accumulating_answer = ee.Image(accumulating_answer)
    one_event = ee.Image(one_event)

    # Assume all values are valid
    one_event_valid_values = one_event

    # Extract prior probabilities and comparison layer from accumulating answer
    current_probs = extract_named_slot(accumulating_answer, probs_label)
    comparison_layer = extract_named_slot(accumulating_answer, comparison_layer_label)

    # Update image counters
    total_image_counter = (
        extract_named_slot(accumulating_answer, total_image_counter_label)
        .add(1)
        .rename([total_image_counter_label])
    )
    accumulating_answer = overwrite_named_slot(
        accumulating_answer, total_image_counter, total_image_counter_label
    )

    if record_iteration_number_at_each_time_step:
        accumulating_answer = accumulating_answer.addBands(
            total_image_counter.rename(["iter"])
        )

    per_pixel_image_counter = extract_named_slot(
        accumulating_answer, per_pixel_image_counter_label
    )
    per_pixel_image_counter = per_pixel_image_counter.where(
        one_event_valid_values, per_pixel_image_counter.add(1)
    ).rename([per_pixel_image_counter_label])
    accumulating_answer = overwrite_named_slot(
        accumulating_answer, per_pixel_image_counter, per_pixel_image_counter_label
    )

    if record_events_at_each_time_step:
        accumulating_answer = accumulating_answer.addBands(
            one_event.mask(one_event_valid_values).rename([event_label])
        )

    # Handle different transition creation methods
    if transition_creation_method == "C":
        transition_array_image_rescaled_and_dampened = (
            create_1d_transition_array_image_from_dynamic_truth(
                one_event,
                truth_table_stack,
                list_of_event_classes_from_0,
                cond_stack_label,
            )
        )

    # Record conditionals if needed
    if record_conditionals_at_each_time_step:
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
    ).rename([probs_label])
    accumulating_answer = accumulating_answer.addBands(
        posterior_probs_all_pixels_1d, [probs_label], True
    )

    if record_probabilities_at_each_time_step:
        accumulating_answer = accumulating_answer.addBands(
            posterior_probs_all_pixels_1d.arrayFlatten([class_name_list])
        )

    # Record BULC layers and confidence if needed
    if record_bulc_layers_at_each_time_step:
        bulc_posterior = (
            posterior_probs_all_pixels_1d.arrayArgmax()
            .arrayFlatten([[bulc_layer_label]])
            .add(1)
        )
        accumulating_answer = accumulating_answer.addBands(
            bulc_posterior, [bulc_layer_label], False
        )

    if record_confidence_at_each_time_step:
        bulc_confidence_1d_array = posterior_probs_all_pixels_1d.arrayReduce(
            ee.Reducer.max(), [0]
        ).select(0)
        bulc_confidence_0d = bulc_confidence_1d_array.reduce(
            ee.Reducer.max()
        ).arrayFlatten([[bulc_confidence_label]])
        accumulating_answer = accumulating_answer.addBands(bulc_confidence_0d)

    # Increment the global image counter
    accumulating_answer = ee.Image(
        increment_global_image_counter(
            accumulating_answer, global_counter_property_name
        )
    )

    return accumulating_answer


def kalman_with_bulcd(args):
    bulc_args = args["kalman_with_bulcd_params"]["bulc_arguments"]
    args = args["kalman_with_bulcd_params"]

    # Parse the parameter dictionary
    events_as_image_collection = ee.ImageCollection(
        bulc_args.get("kalman_with_bulcd_params")
    )
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

    # Setup flags and labels
    record_iteration_number_at_each_time_step = bulc_args.get(
        "record_iteration_number_at_each_time_step"
    )
    record_events_at_each_time_step = bulc_args.get("record_events_at_each_time_step")
    record_probabilities_at_each_time_step = bulc_args.get(
        "record_probabilities_at_each_time_step"
    )
    record_conditionals_at_each_time_step = bulc_args.get(
        "record_conditionals_at_each_time_step"
    )
    record_bulc_layers_at_each_time_step = bulc_args.get(
        "record_bulc_layers_at_each_time_step"
    )
    record_confidence_at_each_time_step = bulc_args.get(
        "record_confidence_at_each_time_step"
    )
    record_final_class = bulc_args.get("record_final_class")

    total_image_counter_label = (
        "total_image_counter"  # Increases anytime a new image is processed
    )
    per_pixel_image_counter_label = "per_pixel_image_counter"  # Increases if a new image is processed and has a valid value in that pixel
    bulc_confidence_label = "BULC_confidence"
    bulc_layer_label = "BULC_classification"

    probs_selector = "prob_cls"
    probs_label = "probability_array"

    cond_stack_label = "conditional_stack"

    comparison_layer_label = "comparison_layer"
    event_label = "event"

    global_counter_property_name = "number_of_images_so_far"

    list_of_event_classes_from_0 = ee.List.sequence(0, max_class_in_event_images)
    list_of_tracked_classes_from_1 = ee.List.sequence(1, number_of_classes_to_track)

    class_name_list = list_of_tracked_classes_from_1.map(
        lambda x: ee.String(probs_selector).cat(ee.String(ee.Number(x).int()))
    )

    # Create an initialization dictionary for the iterate package
    iidd = {
        "initializing_leveler": initializing_leveler,
        "transition_creation_method": transition_creation_method,
        "number_of_classes_to_track": number_of_classes_to_track,
        "default_study_area": default_study_area,
        "class_name_list": class_name_list,
        "probs_label": probs_label,
        "comparison_layer_label": comparison_layer_label,
        "total_image_counter_label": total_image_counter_label,
        "per_pixel_image_counter_label": per_pixel_image_counter_label,
        "cond_stack_label": cond_stack_label,
        "record_probabilities_at_each_time_step": record_probabilities_at_each_time_step,
        "initialization_approach": initialization_approach,
        "base_land_cover_image": (
            base_land_cover_image if initialization_approach == "F" else None
        ),
        "first_comparison_image": bulc_args.get("first_comparison_image"),
        "global_counter_property_name": global_counter_property_name,
        "record_bulc_layers_at_each_time_step": record_bulc_layers_at_each_time_step,
        "record_confidence_at_each_time_step": record_confidence_at_each_time_step,
        "bulc_confidence_label": bulc_confidence_label,
        "bulc_layer_label": bulc_layer_label,
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

    def bulc_binding(image, state):
        return hidden_bulc_iterate_with_options(
            image,
            state,
            probs_label,
            event_label,
            comparison_layer_label,
            total_image_counter_label,
            per_pixel_image_counter_label,
            record_iteration_number_at_each_time_step,
            record_events_at_each_time_step,
            record_conditionals_at_each_time_step,
            truth_table_stack,
            transition_creation_method,
            list_of_event_classes_from_0,
            posterior_leveler,
            posterior_minimum,
            number_of_classes_to_track,
            record_probabilities_at_each_time_step,
            class_name_list,
            transition_leveler,
            global_counter_property_name,
            record_bulc_layers_at_each_time_step,
            record_confidence_at_each_time_step,
            bulc_layer_label,
            bulc_confidence_label,
            cond_stack_label,
            record_final_class,
        )

    the_bound_iterate = ee.Image(
        events_as_image_collection.iterate(bulc_binding, accumulating_answer)
    )

    if record_confidence_at_each_time_step:
        # Get the Probability layers and put them in their own image
        layer_match_key = ".*" + bulc_confidence_label + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_confidence_layers = all_matched_layers_as_multiband

    if record_events_at_each_time_step:
        # Get the Event layers and put them in their own image
        layer_match_key = ".*" + event_label + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_event_layers = all_matched_layers_as_multiband

    if record_bulc_layers_at_each_time_step:
        # Get the BULC layers and put them in their own image
        layer_match_key = ".*" + bulc_layer_label + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_bulc_layers = all_matched_layers_as_multiband

    if record_probabilities_at_each_time_step:
        # Get the Probability layers and put them in their own image
        layer_match_key = ".*" + probs_selector + ".*"

        all_matched_layers_as_multiband = the_bound_iterate.select(layer_match_key)
        all_probability_layers = all_matched_layers_as_multiband

    if record_final_class:
        # Grab the final BULC probabilities
        final_bulc_probs = (
            the_bound_iterate.select(0)
            .arrayFlatten([class_name_list])
            .rename(class_name_list)
        )

    return {
        "multi_band_bulc_return": the_bound_iterate.clip(default_study_area),
        "all_confidence_layers": all_confidence_layers.clip(default_study_area),
        "all_event_layers": all_event_layers.selfMask().clip(default_study_area),
        "all_bulc_layers": all_bulc_layers.selfMask().clip(default_study_area),
        "all_probability_layers": all_probability_layers.clip(default_study_area),
        "final_bulc_probs": final_bulc_probs.clip(default_study_area),
        "default_study_area": default_study_area,
    }
