import ee

from utils.ee.binning import get_one_z_bin
from utils.ee.image_compression_expansion import (
    convert_multi_band_image_to_image_collection,
)

from utils.ee.gather_collections import (
    gather_collections_and_reduce,
)
from utils.ee.harmonic_utils import (
    add_harmonic_bands_via_modality_dictionary,
    determine_harmonic_independents_via_modality_dictionary,
    fit_harmonic_to_collection,
    apply_harmonic_to_collection,
)
from kalman_with_bulcd.parameters import advanced_bulc_parameters


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


def merge_bands_of_images_of_two_collections(collection1, collection2, band_names):
    col1_list = collection1.toList(collection1.size())
    col2_list = collection2.toList(collection2.size())

    def merge_images(index, ic):
        img1 = ee.Image(col1_list.get(index))
        img2 = ee.Image(col2_list.get(index))

        return ee.ImageCollection(ic).merge(
            ee.ImageCollection.fromImages(
                [
                    img1.addBands(img2)
                    .rename(band_names)
                    .copyProperties(img1, ["system:time_start"])
                ]
            )
        )

    return ee.List.sequence(0, col1_list.size().subtract(1)).iterate(
        merge_images,
        ee.ImageCollection.fromImages([]),
    )


def organize_inputs(params):
    """Core function that orchestrates the input preparation process."""
    expectation_collection_parameters = params["expectation_collection_parameters"]
    target_collection_parameters = params["target_collection_parameters"]
    band_name_to_fit = params["band_name_to_fit"]
    default_study_area = expectation_collection_parameters["default_study_area"]

    expectation_collection = gather_collections_and_reduce(
        expectation_collection_parameters
    )

    print(expectation_collection.first().bandNames().getInfo())

    target_collection = gather_collections_and_reduce(target_collection_parameters)

    # Harmonic analysis on expectation collection
    harmonic_independents = determine_harmonic_independents_via_modality_dictionary(
        params["modality_dictionary"]
    )

    expectation_collection = add_harmonic_bands_via_modality_dictionary(
        expectation_collection, params["modality_dictionary"]
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

    print(expectation_collection_fit.getInfo())

    expectation_residuals = expectation_collection_fit.map(
        lambda img: img.select(band_name_to_fit)
        .subtract(img.select(["fitted"]))
        .rename("expectation_difference_optical")
        .copyProperties(img, ["system:time_start"])
    )

    expectation_period_standard_deviation = ee.Image(
        summarize_image_collection_simple(
            expectation_residuals,
            summarization_method="StdDev",
            band_name="expectation_difference_optical",
        )
    )

    # Applying model to target collection
    target_collection = add_harmonic_bands_via_modality_dictionary(
        target_collection, params["modality_dictionary"]
    )

    print(target_collection.size().getInfo())

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

    rescaled_residuals = target_collection_lack_of_fit.multiply(
        params["sensitivity_dictionary"]["z_score_numerator_factor"]
    )

    target_lack_of_fit_as_z_score = (
        ee.Image(rescaled_residuals)
        .divide(
            ee.Image(expectation_period_standard_deviation).max(
                params["sensitivity_dictionary"]["z_score_denominator_factor"]
            )
        )
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

    events_as_image_collection = ee.ImageCollection(
        convert_multi_band_image_to_image_collection(target_lack_of_fit_as_z_score)
    ).map(get_one_z_bin(params["bin_cuts"]))

    kalman_with_bulcd_params = advanced_bulc_parameters()

    kalman_with_bulcd_params["bulc_arguments"] = ee.Dictionary(
        kalman_with_bulcd_params["bulc_arguments"]
    ).set("events_as_image_collection", events_as_image_collection)

    kalman_with_bulcd_params["bulc_arguments"] = ee.Dictionary(
        kalman_with_bulcd_params["bulc_arguments"]
    ).set("default_study_area", default_study_area)

    kalman_with_bulcd_params["kalman_params"] = params["kalman_params"]

    print(target_collection.select(params["band_name_to_fit"]).size().getInfo())

    kalman_with_bulcd_params["events_and_measurements"] = (
        merge_bands_of_images_of_two_collections(
            target_collection.select(params["band_name_to_fit"]),
            events_as_image_collection,
            [params["band_name_to_fit"], ee.String("Slot")],
        )
    )

    return {
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
        "kalman_with_bulcd_params": kalman_with_bulcd_params,
    }
