import ee


def which_bin_internal(val_image, bin_cuts):
    """
    Determines the bin index for each pixel of an image based on specified bin boundaries.

    Args:
    - val_image: ee.Image, the image whose pixel values are to be binned.
    - bin_cuts: list, the boundaries defining the bins.

    Returns:
    - ee.Image, containing the 1-based index of the bin for each pixel.
    """
    # Prepare bin boundaries for 'greater than' and 'less than or equal to' comparisons
    bin_array_for_gt = [-1000000] + bin_cuts  # Prepend a very small number
    bin_array_for_lt = bin_cuts + [1000000]  # Append a very large number

    # Create comparison images
    clause1 = ee.Image(val_image).gt(bin_array_for_gt)
    clause2 = ee.Image(val_image).lte(bin_array_for_lt)

    # Perform a logical 'and' between the two comparison results
    the_anded = clause1.And(clause2).toArray()

    # Determine the index of the highest true value (i.e., the bin each pixel falls into)
    which_slot = (
        the_anded.arrayArgmax().arrayFlatten([["Slot"]]).add(1)
    )  # Adjust for 1-based index

    return which_slot


def get_one_z_bin(bin_cuts):
    """
    Returns a function that, given a value image, returns the bin it belongs to according to specified bin cuts.

    Args:
    - bin_cuts: Dictionary with keys 'bin_array_for_gt' and 'bin_array_for_lt', the thresholds for binning.

    Returns:
    - Function that takes an ee.Image and returns an ee.Image representing the bin number.
    """

    def inner(image):
        which_bin = which_bin_internal(image, bin_cuts)
        which_bin = which_bin.copyProperties(image, ["system:time_start"])
        return which_bin

    return inner
