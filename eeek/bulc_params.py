import ee


def get_bulc_parameter_dictionary():
    # Initialize parameters
    initializing_leveler = 0.7
    transition_leveler = 0.7
    posterior_leveler = 0.9
    number_of_classes_in_events = 10
    number_of_classes_to_track = 3
    base_land_cover_image = ee.Image.constant(2)  # Default: "nothing has changed"
    first_comparison_image = base_land_cover_image  # Starting with the base image

    # Recording flags
    record_iteration_number = False
    record_events = False
    record_conditionals = False
    record_probabilities = False
    record_bulc_layers = True
    record_confidence = False
    record_final_class = True
    record_final_probabilities = True
    verbose_stack = False

    # Initialization and transition approaches
    initialization_approach = "F"  # F: first run, as opposed to continuing run
    overlay_approach = "C"  # C: Custom matrix, D: Identity matrix, V: Overlay

    # Define custom transition matrix
    if overlay_approach == "C":
        custom_transition_matrix = [
            [0.83, 0.08, 0.08],
            [0.66, 0.24, 0.08],
            [0.53, 0.37, 0.08],
            [0.14, 0.76, 0.08],
            [0.08, 0.83, 0.08],
            [0.08, 0.83, 0.08],
            [0.08, 0.76, 0.14],
            [0.08, 0.37, 0.53],
            [0.08, 0.24, 0.66],
            [0.08, 0.08, 0.83],
        ]

    if overlay_approach == "V":
        number_of_stratified_points = 1000
        stratum_image = ee.Image(1)  # Placeholder for an actual image

    # Prepare BULC argument dictionary
    bulc_argument_dict = {
        "initializing_leveler": initializing_leveler,
        "transition_leveler": transition_leveler,
        "transition_minimum": (1 - transition_leveler) / number_of_classes_to_track,
        "posterior_leveler": posterior_leveler,
        "posterior_minimum": (1 - posterior_leveler) / number_of_classes_to_track,
        "max_class_in_event_images": number_of_classes_in_events,
        "number_of_classes_to_track": number_of_classes_to_track,
        "record_iteration_number_at_each_time_step": record_iteration_number,
        "record_events_at_each_time_step": record_events,
        "record_probabilities_at_each_time_step": record_probabilities,
        "record_conditionals_at_each_time_step": record_conditionals,
        "record_bulc_layers_at_each_time_step": record_bulc_layers,
        "record_confidence_at_each_time_step": record_confidence,
        "record_final_class": record_final_class,
        "record_final_probabilities": record_final_probabilities,
        "custom_transition_matrix": (
            custom_transition_matrix if overlay_approach == "C" else None
        ),
        "number_of_stratified_points_for_overlay": (
            number_of_stratified_points if overlay_approach == "V" else None
        ),
        "stratum_image": stratum_image if overlay_approach == "V" else None,
    }

    return {
        "BULCargumentDictionary": bulc_argument_dict,
        "initialization_approach": initialization_approach,
        "transition_creation_method": overlay_approach,
        "base_land_cover_image": base_land_cover_image,
        "first_comparison_image": first_comparison_image,
    }
