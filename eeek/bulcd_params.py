import ee

# Define coordinates and create a default study area
coordinates = [(-126.04, 49.59), (-126.04, 40.76), (-118.93, 40.76), (-118.93, 49.59)]

default_study_area = ee.Geometry.Polygon(coordinates, None, False)

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

# Prepare the main arguments input dictionary
bulcd_params = {
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
}
