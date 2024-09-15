import ee

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

from utils import utils
from bulc.bulc_d import run_bulc_d_algorithm
from bulc.organize_bulcd_inputs import organize_bulcd_inputs
from bulc.bulcd_params import bulcd_params
from bulc.bulc_params import get_bulc_parameter_dictionary
from pprint import pprint

coords = (-122.48489, 45.079215)
point = ee.Geometry.Point(coords)

geometries = ee.FeatureCollection(ee.List([ee.Feature(point)]))


def run_bulcd():
    organized_inputs = organize_bulcd_inputs(bulcd_params)
    bulc_d_input = dict()

    bulc_d_input["default_study_area"] = bulcd_params["default_study_area"]
    bulc_d_input["bin_cuts"] = bulcd_params["bin_cuts"]
    bulc_d_input["target_lack_of_fit_as_z_score"] = organized_inputs[
        "target_lack_of_fit_as_z_score"
    ]
    bulc_d_input["bulc_params"] = get_bulc_parameter_dictionary()

    output = run_bulc_d_algorithm(bulc_d_input)
    # image = output["target_lack_of_fit_as_z_score"].clip(point)
    # image = output["final_bulc_probs"].clip(point)

    # for title, data in organized_inputs.items():
    #     print(f"\n\n")
    #     try:
    #         request = utils.build_request(coords)
    #         request["expression"] = data
    #         result = utils.compute_pixels_wrapper(request)

    #         pprint(f"{title}: {result.tolist()}")
    #     except Exception as e:
    #         # print(e)
    #         try:
    #             request = utils.build_request(coords)
    #             request["expression"] = data.mosaic()
    #             result = utils.compute_pixels_wrapper(request)
    #             pprint(f"{title}: {result}")
    #         except Exception as e:
    #             try:
    #                 pprint(f"{title}: {data.getInfo()}")
    #             except Exception as e:
    #                 continue


if __name__ == "__main__":
    run_bulcd()
