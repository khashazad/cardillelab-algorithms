import math
import ee


def get_rmse_and_r2(linear_trend, image_collection, independents, dependent):
    image_collection = ee.ImageCollection(image_collection)
    n = ee.ImageCollection(image_collection).select(dependent).count()
    dof = n.subtract(independents.length())
    rmsr = linear_trend.select("residuals").arrayProject([0]).arrayFlatten([["rmsr"]])
    rss = rmsr.pow(2).multiply(n)
    s_squared = rss.divide(dof)
    y_variance = image_collection.select(dependent).reduce(ee.Reducer.sampleVariance())
    r_square_adj = ee.Image(1).subtract(s_squared.divide(y_variance))

    return {
        "the_residuals": rmsr,
        "the_rmse": rmsr,
        "the_r2": r_square_adj,
    }


def fit_harmonic_to_collection(harmonic_ic, tracked_band, harmonic_independents):
    harmonic_ic = ee.ImageCollection(harmonic_ic)
    dependent = ee.String(tracked_band)
    harmonic_independents = ee.List(harmonic_independents)
    harmonic_trend = harmonic_ic.select(harmonic_independents.add(dependent)).reduce(
        ee.Reducer.linearRegression(harmonic_independents.length(), 1)
    )
    harmonic_trend_coefficients = (
        harmonic_trend.select("coefficients")
        .arrayProject([0])
        .arrayFlatten([harmonic_independents])
    )

    rmse_and_r2 = get_rmse_and_r2(
        harmonic_trend, harmonic_ic, harmonic_independents, tracked_band
    )
    the_return = {}
    the_return["harmonic_trend_coefficients"] = harmonic_trend_coefficients
    the_return["the_residuals"] = rmse_and_r2["the_residuals"]
    the_return["the_r2"] = rmse_and_r2["the_r2"]
    the_return["harmonic_trend_coefficients"] = harmonic_trend_coefficients
    the_return["linear_trend_object"] = harmonic_trend
    return the_return


def apply_harmonic_to_collection(
    harmonic_ic, tracked_band, harmonic_independents, harmonic_trend_coefficients
):
    harmonic_ic = ee.ImageCollection(harmonic_ic)
    dependent = ee.String(tracked_band)
    harmonic_independents = ee.List(harmonic_independents)

    print(harmonic_trend_coefficients.getInfo())    

    fitted_harmonic = harmonic_ic.map(
        lambda image: image.addBands(
            image.select(harmonic_independents)
            .multiply(harmonic_trend_coefficients)
            .reduce("sum")
            .rename("fitted")
        )
    )

    print(fitted_harmonic.first().bandNames().getInfo())
    return fitted_harmonic


def add_harmonic_bands_via_modality_dictionary(image_collection, modality_dictionary):
    vm = modality_dictionary  # for convenience of typing
    if (
        vm["bimodal"]
        or vm["constant"]
        or vm["linear"]
        or vm["trimodal"]
        or vm["unimodal"]
    ):
        args = {}
        args["date_property"] = "frac_doy"
        args["time_band_name"] = "t"
        reduced_collection_with_harmonic = add_time_band_using_property_name_to_ic(
            image_collection, args
        )
        if vm["constant"]:
            reduced_collection_with_harmonic = add_linear_constant_to_ic(
                reduced_collection_with_harmonic, 1, "constant"
            )
        if vm["unimodal"]:
            var_args_harmonic_bands = {}
            var_args_harmonic_bands["time_band_name"] = args["time_band_name"]
            var_args_harmonic_bands["non_standard_period_boolean"] = False
            var_args_harmonic_bands["cos_name"] = "cos"
            var_args_harmonic_bands["sin_name"] = "sin"

            reduced_collection_with_harmonic = add_cos_and_sin_bands_to_collection(
                reduced_collection_with_harmonic, var_args_harmonic_bands
            )
        if vm["bimodal"]:
            var_args_harmonic_bands = {}
            var_args_harmonic_bands["time_band_name"] = args["time_band_name"]
            var_args_harmonic_bands["non_standard_period_boolean"] = True
            var_args_harmonic_bands["non_standard_period_value"] = 1
            var_args_harmonic_bands["cos_name"] = "cos2"
            var_args_harmonic_bands["sin_name"] = "sin2"

            reduced_collection_with_harmonic = add_cos_and_sin_bands_to_collection(
                reduced_collection_with_harmonic, var_args_harmonic_bands
            )
        if vm["trimodal"]:
            var_args_harmonic_bands = {}
            var_args_harmonic_bands["time_band_name"] = args["time_band_name"]
            var_args_harmonic_bands["non_standard_period_boolean"] = True
            var_args_harmonic_bands["non_standard_period_value"] = 0.667
            var_args_harmonic_bands["cos_name"] = "cos3"
            var_args_harmonic_bands["sin_name"] = "sin3"

            reduced_collection_with_harmonic = add_cos_and_sin_bands_to_collection(
                reduced_collection_with_harmonic, var_args_harmonic_bands
            )
    return reduced_collection_with_harmonic


def determine_harmonic_independents_via_modality_dictionary(modality_dictionary):
    vm = modality_dictionary
    harmonic_list = []
    if vm["constant"]:
        harmonic_list.append("constant")
    if vm["linear"]:
        harmonic_list.append("t")
    if vm["unimodal"]:
        harmonic_list.append("cos")
        harmonic_list.append("sin")
    if vm["bimodal"]:
        harmonic_list.append("cos2")
        harmonic_list.append("sin2")
    if vm["trimodal"]:
        harmonic_list.append("cos3")
        harmonic_list.append("sin3")
    return harmonic_list


def add_time_band_using_property_name_to_ic(an_ic, args):
    date_property = args["date_property"]
    time_band_name = args["time_band_name"]

    def add_fractional_year_from_property_name(an_image):
        number_for_each_pixel = ee.Number(an_image.get(date_property))
        the_time_band = ee.Image(number_for_each_pixel).rename(time_band_name).float()
        return an_image.addBands(the_time_band)

    the_ic_return = ee.ImageCollection(an_ic).map(
        add_fractional_year_from_property_name
    )
    return the_ic_return


def add_linear_constant_to_ic(an_ic, constant_value, constant_band_name):
    def add_constant_to_img(img):
        return img.addBands(ee.Image(constant_value).rename(constant_band_name))

    return an_ic.map(add_constant_to_img)


def add_cos_and_sin_bands_to_collection(collection, var_args_harmonic_bands):
    time_band_name = var_args_harmonic_bands["time_band_name"]
    non_standard_period_boolean = var_args_harmonic_bands["non_standard_period_boolean"]

    cos_name = var_args_harmonic_bands["cos_name"]
    sin_name = var_args_harmonic_bands["sin_name"]

    if non_standard_period_boolean:
        non_standard_period_value = var_args_harmonic_bands["non_standard_period_value"]
        period = non_standard_period_value
    else:
        period = 2

    angular_frequency = ee.Number(2).divide(period)
    two_pi_freq = angular_frequency.multiply(2 * math.pi)

    def get_cos_and_sin_for_date_given_angular_freq(an_image):
        the_time_band = ee.Image(an_image).select(time_band_name)
        time_radians = the_time_band.multiply(two_pi_freq)
        the_cos_factor = time_radians.cos().rename(cos_name)
        the_sin_factor = time_radians.sin().rename(sin_name)
        return an_image.addBands(the_cos_factor).addBands(the_sin_factor)

    the_ic_return = ee.ImageCollection(collection).map(
        get_cos_and_sin_for_date_given_angular_freq
    )
    return the_ic_return


def add_harmonic_independents_using_property_name_to_ic(
    an_ic,
    the_frac_doy_property,
    the_day_range_property,
    non_standard_period_boolean,
    non_standard_period_value,
):
    if non_standard_period_boolean:
        period = non_standard_period_value
    else:
        period = 2
    inverse_period = 1 / period
    angular_frequency = 2 * inverse_period
    angular_frequency_image = ee.Image(angular_frequency).rename("angular_freq").float()

    def add_fractional_year_from_property_name(an_image):
        day_range = ee.Number(an_image.get(the_day_range_property))
        number_for_each_pixel = ee.Number(an_image.get(the_frac_doy_property))
        the_time_band = ee.Image(number_for_each_pixel).rename("t").float()
        time_radians = the_time_band.multiply(angular_frequency_image)
        the_cos_factor = time_radians.cos().rename("cos")
        the_sin_factor = time_radians.sin().rename("sin")
        the_ang_freq = angular_frequency_image.rename("angular_freq")
        flat_radians = (
            number_for_each_pixel.multiply(angular_frequency)
            .multiply(2)
            .multiply(math.pi)
        )
        a_cos_number = flat_radians.cos()
        an_image = an_image.set("image_cosine", a_cos_number)
        a_sin_number = flat_radians.sin()
        an_image = an_image.set("image_sine", a_sin_number)
        an_image = an_image.set("image_angular_frequency", angular_frequency)
        return (
            an_image.addBands(ee.Image.constant(1))
            .addBands(the_time_band)
            .addBands(the_cos_factor)
            .addBands(the_sin_factor)
        )

    the_ic_return = ee.ImageCollection(an_ic).map(
        add_fractional_year_from_property_name
    )
    return the_ic_return


def compute_fit_and_get_coefficients(an_ic, band_name_to_fit, harmonic_constant):
    dependent = band_name_to_fit
    harmonic_independents = ["constant", "t", "cos", "sin"]
    harmonic_coefficient_names = ["β0", "β1", "β2", "β3"]
    harmonic_imagery = ee.ImageCollection(an_ic)
    independents = harmonic_independents
    coefficients = harmonic_coefficient_names
    if harmonic_constant:
        independents = ["constant"]
        coefficients = ["β0"]
    harmonic_trend = harmonic_imagery.select(independents + [dependent]).reduce(
        ee.Reducer.linearRegression(len(independents), 1)
    )
    harmonic_trend_coefficients = (
        harmonic_trend.select("coefficients")
        .arrayProject([0])
        .arrayFlatten([coefficients])
    )
    the_regression_elements = {}
    the_regression_elements["harmonic_trend_coefficients"] = harmonic_trend_coefficients
    return harmonic_trend_coefficients
