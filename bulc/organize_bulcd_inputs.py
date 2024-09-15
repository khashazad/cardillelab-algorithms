import ee

from utils.ee.gather_collections import (
    gather_collections_and_reduce,
    check_for_optical,
    check_for_sar,
)
from utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    determine_harmonic_independents_via_modality_dictionary,
    fit_harmonic_to_collection,
    apply_harmonic_to_collection,
)


def summarize_image_collection_simple(
    image_collection, summarization_method, band_name
):
    if summarization_method == "LastNonNull":
        summary_statistic = (
            image_collection.select(band_name)
            .reduce(ee.Reducer.lastNonNull())
            .rename("ICPeriodSummaryValue")
        )
        return summary_statistic
    elif summarization_method == "StdDev":
        summary_statistic = (
            image_collection.select(band_name)
            .reduce(ee.Reducer.sampleStdDev())
            .rename("EstSDofIC")
        )
        return summary_statistic


def difference_observed_vs_fitted(collection, target_band, fitted_band):
    subtracted_collection = collection.map(
        lambda oneImage: oneImage.select(target_band)
        .subtract(oneImage.select(fitted_band))
        .copyProperties(oneImage, ["system:time_start"])
    )
    return subtracted_collection.toBands()


def gathering_args_parking(gathering_collections_args):
    """Deep copy function for arguments dictionary."""
    parked_args = {}
    for key, value in gathering_collections_args.items():
        parked_args[key] = ee.FeatureCollection(value).map(
            lambda feature: feature.set(key, value)
        )

    parked_args["expectation_collection_parameters"]["default_study_area"] = (
        gathering_collections_args["expectation_collection_parameters"][
            "default_study_area"
        ]
    )
    parked_args["target_collection_parameters"]["default_study_area"] = (
        gathering_collections_args["target_collection_parameters"]["default_study_area"]
    )
    return parked_args


def get_first_or_second_or_average_values_across_mb(img1, img2):
    """Function to blend Optical and SAR streams."""
    the_sum = img1.add(img2).divide(2)
    the_answer = the_sum.where(the_sum.eq(0), img1.add(img2))
    return the_answer.updateMask(the_answer.neq(0))


def blend_optical_and_sar_streams(optical_stream, sar_stream):
    """Function to blend Optical and SAR data streams."""
    # Default to optical data for many attributes, focus on blending targetLOFAsZScore
    blended_bulcd_stream = {**optical_stream}  # Start with a copy of the optical stream
    blended_bulcd_stream["target_lof_as_z_score"] = (
        get_first_or_second_or_average_values_across_mb(
            optical_stream["target_lof_as_z_score"], sar_stream["target_lof_as_z_score"]
        )
    )
    return blended_bulcd_stream


def organize_bulcd_inputs(var_args):
    """Core function that orchestrates the input preparation process."""
    expectation_collection_parameters = var_args["expectation_collection_parameters"]
    target_collection_parameters = var_args["target_collection_parameters"]
    band_name_to_fit = var_args["band_name_to_fit"]

    expectation_collection = gather_collections_and_reduce(
        expectation_collection_parameters
    ).filterBounds(expectation_collection_parameters["default_study_area"])
    print("expectation collection size", expectation_collection.size().getInfo())

    target_collection = gather_collections_and_reduce(
        target_collection_parameters
    ).filterBounds(target_collection_parameters["default_study_area"])

    # Harmonic analysis on expectation collection
    harmonic_independents = determine_harmonic_independents_via_modality_dictionary(
        var_args["modality_dictionary"]
    )
    expectation_collection = add_harmonic_bands_via_modality_dictionary(
        expectation_collection, var_args["modality_dictionary"]
    )
    expectation_year_regression = fit_harmonic_to_collection(
        expectation_collection, band_name_to_fit, harmonic_independents
    )
    a_coefficient_set_expectation_year = expectation_year_regression[
        "harmonic_trend_coefficients"
    ]
    the_expectation_r2 = expectation_year_regression["the_r2"]
    expectation_collection_fit = apply_harmonic_to_collection(
        expectation_collection,
        band_name_to_fit,
        harmonic_independents,
        a_coefficient_set_expectation_year,
    )

    expectation_residuals = expectation_collection_fit.map(
        lambda img: img.select(band_name_to_fit)
        .subtract(img.select(["fitted"]))
        .rename("expectation_difference_optical")
        .copyProperties(img, ["system:time_start"])
    )

    expectation_residuals_abs = expectation_residuals.map(
        lambda img: img.abs()
        .rename("expectation_difference_optical_abs_value")
        .copyProperties(img, ["system:time_start"])
    )

    expectation_period_summary_value = ee.Image(
        summarize_image_collection_simple(
            expectation_collection,
            summarization_method="LastNonNull",
            band_name=band_name_to_fit,
        )
    )

    expectation_period_standard_deviation = ee.Image(
        summarize_image_collection_simple(
            expectation_residuals,
            summarization_method="StdDev",
            band_name="expectation_difference_optical",
        )
    )

    print(expectation_period_standard_deviation.getInfo())

    # Applying model to target collection
    target_collection = add_harmonic_bands_via_modality_dictionary(
        target_collection, var_args["modality_dictionary"]
    )
    target_collection_fit = apply_harmonic_to_collection(
        target_collection,
        band_name_to_fit,
        harmonic_independents,
        a_coefficient_set_expectation_year,
    )

    target_residuals = target_collection_fit.map(
        lambda img: img.select(band_name_to_fit)
        .subtract(img.select(["fitted"]))
        .rename("targetDifference")
        .copyProperties(img, ["system:time_start"])
    )

    target_collection_lack_of_fit = difference_observed_vs_fitted(
        target_collection_fit, band_name_to_fit, "fitted"
    )

    target_lack_of_fit_as_z_score = (
        ee.Image(target_collection_lack_of_fit)
        .divide(ee.Image(expectation_period_standard_deviation))
        .max(-10)
    )

    target_period_summary_value = ee.Image(
        summarize_image_collection_simple(
            target_collection,
            summarization_method="LastNonNull",
            band_name=band_name_to_fit,
        )
    )

    if band_name_to_fit == "swir":
        target_lack_of_fit_as_z_score = target_lack_of_fit_as_z_score.multiply(-1)

    organized_bulcd_stream = {
        "expectation_collection": expectation_collection,
        "expectation_year_regression_object": expectation_year_regression,
        "a_coefficient_set_expectation_year": a_coefficient_set_expectation_year,
        "the_expectation_r2": the_expectation_r2,
        "expectation_collection_fit": expectation_collection_fit,
        "expectation_residuals": expectation_residuals,
        "expectation_period_standard_deviation": expectation_period_standard_deviation,
        "target_collection": target_collection,
        "target_collection_fit": target_collection_fit,
        "target_residuals": target_residuals,
        "target_collection_lack_of_fit": target_collection_lack_of_fit,
        "target_lack_of_fit_as_z_score": target_lack_of_fit_as_z_score,
        "target_period_summary_value": target_period_summary_value,
    }

    has_optical = check_for_optical(expectation_collection_parameters)
    has_sar = check_for_sar(expectation_collection_parameters)
    # if has_optical and has_sar:
    #     parked_args = gathering_args_parking(var_args)
    #     sar_stream = create_exp_and_target_stream(
    #         parked_args
    #     )  # assuming this function exists
    #     optical_stream = create_exp_and_target_stream(
    #         var_args
    #     )  # assuming this function exists
    #     organized_bulcd_stream = blend_optical_and_sar_streams(
    #         optical_stream, sar_stream
    #     )

    return organized_bulcd_stream
