import ee

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

from eeek.bulc_d import run_bulc_d_algorithm
from eeek.organize_bulcd_inputs import organize_bulcd_inputs
from eeek.bulcd_params import bulcd_params
from eeek.bulc_params import get_bulc_parameter_dictionary
from pprint import pprint


def run_bulcd():
    organized_inputs = organize_bulcd_inputs(bulcd_params)
    bulc_d_input = dict()

    bulc_d_input["default_study_area"] = bulcd_params["default_study_area"]
    bulc_d_input["bin_cuts"] = bulcd_params["bin_cuts"]
    bulc_d_input["target_lack_of_fit_as_z_score"] = organized_inputs[
        "target_lack_of_fit_as_z_score"
    ]
    bulc_d_input["bulc_params"] = get_bulc_parameter_dictionary()

    run_bulc_d_algorithm(bulc_d_input)


if __name__ == "__main__":
    run_bulcd()
