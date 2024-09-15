import ee
from utils.ee.image_compression_expansion import (
    convert_multi_band_image_to_image_collection,
)
from utils.ee.binning import get_one_z_bin
from bulc.bulc import bulc
from pprint import pprint


def run_bulc_d_algorithm(args):
    bulc_params = args["bulc_params"]
    events_as_image_collection = ee.ImageCollection(
        convert_multi_band_image_to_image_collection(
            args["target_lack_of_fit_as_z_score"]
        )
    ).map(get_one_z_bin(args["bin_cuts"]))

    bin_average = events_as_image_collection.reduce(ee.Reducer.mean())

    bulc_params["bulc_arguments"] = ee.Dictionary(bulc_params["bulc_arguments"]).set(
        "events_as_image_collection", events_as_image_collection
    )

    bulc_params["bulc_arguments"] = ee.Dictionary(bulc_params["bulc_arguments"]).set(
        "default_study_area", args["default_study_area"]
    )

    bulc_output = bulc(bulc_params)

    theFinalBULCprobs = ee.Image(bulc_output["final_bulc_probs"])

    return {
        "all_bulc_layers": bulc_output["all_bulc_layers"],
        "all_confidence_layers": bulc_output["all_confidence_layers"],
        "all_event_layers": bulc_output["all_event_layers"],
        "all_probability_layers": bulc_output["all_probability_layers"],
        "final_bulc_probs": bulc_output["final_bulc_probs"],
        "multi_band_bulc_return": bulc_output["multi_band_bulc_return"],
        "target_lack_of_fit_as_z_score": args["target_lack_of_fit_as_z_score"],
        "events_as_image_collection": events_as_image_collection,
    }
