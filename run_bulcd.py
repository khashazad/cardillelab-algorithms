import ee

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

from eeek.organize_bulcd_inputs import organize_bulcd_inputs
from eeek.bulcd_params import bulcd_params
from eeek.bulc_params import get_bulc_parameter_dictionary
from pprint import pprint


def run_bulcd():
    organized_inputs = organize_bulcd_inputs(bulcd_params)
    bulc_params = get_bulc_parameter_dictionary()

    # pprint(bulcd_params)


if __name__ == "__main__":
    run_bulcd()
