import ee


def add_dummy_row78(an_array):
    # Convert the input to an Earth Engine array
    array = ee.Array(ee.List(an_array))
    # Get the length of the array
    lengths = array.length()

    # Create a 1x* array filled with ones
    ones = ee.Array([ee.List.repeat(1, lengths.get([1]))])

    # Concatenate the ones array with the original array along the first dimension (0)
    catted = ee.Array.cat([ones, array], 0)

    return catted


def dampen_truth_and_add_dummy_row11(
    truth_table, transition_leveler, transition_minimum
):
    # Dampen the truth table using the transition leveler and transition minimum
    damp_truth_table = (
        ee.Array(truth_table).multiply(transition_leveler).add(transition_minimum)
    )

    damp_with_dummy = add_dummy_row78(damp_truth_table)

    return damp_with_dummy
