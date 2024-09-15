import ee


def advanced_bulc_parameters():
    # Initialize parameters
    parameters = {
        "initializing_leveler": 0.7,
        "transition_leveler": 0.7,
        "posterior_leveler": 0.9,
        "number_of_classes_in_events": 10,
        "number_of_classes_to_track": 3,
        "base_land_cover_image": ee.Image.constant(2),
        "first_comparison_image": ee.Image.constant(2),
        "record_iteration_number": False,
        "record_events": False,
        "record_conditionals": False,
        "record_probabilities": False,
        "record_bulc_layers": True,
        "record_confidence": False,
        "record_final_class": True,
        "record_final_probabilities": True,
        "verbose_stack": False,
        "initialization_approach": "F",
        "overlay_approach": "C",
        "custom_transition_matrix": (
            [
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
            if "C"
            else None
        ),
        "number_of_stratified_points_for_overlay": 1000 if "V" else None,
        "stratum_image": ee.Image(1) if "V" else None,
    }

    return {
        "bulc_arguments": parameters,
        "initialization_approach": parameters["initialization_approach"],
        "transition_creation_method": parameters["overlay_approach"],
        "base_land_cover_image": parameters["base_land_cover_image"],
        "first_comparison_image": parameters["first_comparison_image"],
    }


def run_specific_parameters(study_area=None):
    # Define coordinates and create a default study area
    coordinates = [
        (-126.04, 49.59),
        (-126.04, 40.76),
        (-118.93, 40.76),
        (-118.93, 49.59),
    ]
    default_study_area = (
        study_area if study_area else ee.Geometry.Polygon(coordinates, None, False)
    )

    # Define dataset selections
    dataset_selection = {
        "L5": False,
        "L7": False,
        "L8": True,
        "L9": True,
        "MO": False,
        "S2": False,
        "S1": False,
        "DW": False,
        "AL": False,
        "NI": False,
    }

    # Define reduction method and band name
    which_reduction = "SWIR"
    band_name = "swir"
    day_step_size = 4

    # Expectation collection parameters
    expectation_collection_parameters = {
        "which_reduction": which_reduction,
        "band_name_reduction": band_name,
        "default_study_area": default_study_area,
        "day_step_size": day_step_size,
        "dataset_selection": dataset_selection,
        "L5dictionary": {
            "years_list": [2021],
            "first_doy": 1,
            "last_doy": 90,
            "cloud_cover_threshold": 45,
        },
        "L7dictionary": {
            "years_list": [2021],
            "first_doy": 1,
            "last_doy": 90,
            "cloud_cover_threshold": 47,
        },
        "L8dictionary": {
            "years_list": [2022],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2022],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "Modictionary": {
            "years_list": [1950],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 40,
        },
        "S2dictionary": {
            "years_list": [2021],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 70,
            "s2cloudless": {
                "cloud_cover_threshold": 70,
                "cld_prb_thresh": 50,
                "nir_drk_thresh": 0.15,
                "cld_prj_dist": 3,
                "buffer": 50,
                "aoi": default_study_area,
            },
        },
        "S1dictionary": {
            "sar_value_to_track": "HV",
            "is_radar": True,
            "years_list": [2021],
            "first_doy": 270,
            "last_doy": 360,
        },
    }

    # Target collection parameters
    target_collection_parameters = {
        "which_reduction": which_reduction,
        "default_study_area": default_study_area,
        "day_step_size": day_step_size,
        "band_name_reduction": band_name,
        "dataset_selection": dataset_selection,
        "L5dictionary": {
            "years_list": [2022],
            "first_doy": 115,
            "last_doy": 165,
            "cloud_cover_threshold": 15,
        },
        "L7dictionary": {
            "years_list": [2022],
            "first_doy": 117,
            "last_doy": 167,
            "cloud_cover_threshold": 17,
        },
        "L8dictionary": {
            "years_list": [2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "Modictionary": {
            "years_list": [2022],
            "first_doy": 110,
            "last_doy": 210,
            "cloud_cover_threshold": 10,
        },
        "S2dictionary": {
            "years_list": [2022],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 70,
            "s2cloudless": {
                "cloud_cover_threshold": 70,
                "cld_prb_thresh": 50,
                "nir_drk_thresh": 0.15,
                "cld_prj_dist": 3,
                "buffer": 50,
                "aoi": default_study_area,
            },
        },
        "S1dictionary": {
            "sar_value_to_track": "HV",
            "is_radar": True,
            "years_list": [2023],
            "first_doy": 1,
            "last_doy": 365,
        },
    }
    return {
        "which_reduction": which_reduction,
        "band_name_reduction": band_name,
        "band_name_to_fit": band_name,
        "plotting_means": True,
        "harmonic_constant": False,
        "bin_cuts": [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        "modality_dictionary": {
            "bimodal": False,
            "constant": True,
            "linear": False,
            "trimodal": False,
            "unimodal": True,
        },
        "default_study_area": default_study_area,
        "expectation_collection_parameters": expectation_collection_parameters,
        "target_collection_parameters": target_collection_parameters,
        "kalman_params": {
            "Q": [0.00125, 0.000125, 0.000125],
            "R": 0.003,
            "P": [0.00101, 0.00222, 0.00333],
            "change_probability_threshold": 0.65,
            "Q_scale_factor": 10.0,
        },
    }
