import ee
import pandas as pd
import csv
from pprint import pprint

# Initialize the Earth Engine library.
ee.Initialize()


# Function to mask pixels with low CS+ QA scores.
def mask_low_qa(image):
    qa_band = "cs"
    clear_threshold = 0.60
    mask = image.select(qa_band).gte(clear_threshold)
    return image.updateMask(mask)


# Define the cloud mask function for Landsat 8 and 9.
def mask_sr_clouds_l8_and_l9(image):
    qa_mask = image.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
    saturation_mask = image.select("QA_RADSAT").eq(0)
    return image.updateMask(qa_mask).updateMask(saturation_mask)


def cloud_mask_ic_l5_and_l7(input_image):
    qa_band_name_input = "QA_PIXEL"

    def afn_cloud_masked_nested_scope(input_image):
        input_image = ee.Image(input_image)
        cloud_shadow_bit_mask = 1 << 3
        clouds_bit_mask = 1 << 4
        qa = input_image.select(qa_band_name_input)
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        )
        return input_image.updateMask(mask).copyProperties(
            input_image, ["system:time_start"]
        )

    return afn_cloud_masked_nested_scope(input_image)


def cloud_mask_ic_mo(input_image):
    return input_image


def cloud_mask_ic_older_s2(input_image):
    def afn_cloud_masked_nested_scope(input_image, params):
        input_image = ee.Image(input_image)
        cloud_bqa = input_image.select(params["qa_band_name_input"])
        cloud_mask = (
            cloud_bqa.bitwiseAnd(params["cloud_bit"])
            .eq(params["clouds_bit_thresh"])
            .And(
                cloud_bqa.bitwiseAnd(params["cirrus_bit"]).eq(
                    params["cirrus_bit_thresh"]
                )
            )
            .And(
                cloud_bqa.bitwiseAnd(params["water_bit"]).eq(params["water_bit_thresh"])
            )
        )
        return input_image.updateMask(cloud_mask)

    params = {
        "qa_band_name_input": "QA60",
        "cloud_bit": 2**10,
        "cirrus_bit": 2**11,
        "water_bit": 2**11,
        "clouds_bit_thresh": 0,
        "cirrus_bit_thresh": 0,
        "water_bit_thresh": 0,
    }

    return afn_cloud_masked_nested_scope(input_image, params)


# Date functions
def date_from_day(year, day):
    start_date = ee.Date.fromYMD(year, 1, 1)
    return start_date.advance(day - 1, "day")


def doy_to_month(doy, the_year):
    date_in_year = date_from_day(the_year, doy)
    return ee.Number.parse(date_in_year.format("M")).getInfo()


def doy_to_day(doy, the_year):
    date_in_year = date_from_day(the_year, doy)
    return ee.Number.parse(date_in_year.format("d")).getInfo()


# Scaling functions
def apply_scale_factors_l89(image):
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


def apply_scale_factors_s2(image):
    return image.addBands(image.select("B.").multiply(0.0001))


# Check functions
def check_for_sar(collecting_params):
    the_sar_answer = False
    if (
        collecting_params["dataset_selection"]["S1"]
        or collecting_params["dataset_selection"]["AL"]
        or collecting_params["dataset_selection"]["NI"]
    ):
        the_sar_answer = True
    return the_sar_answer


def check_for_optical(collecting_params):
    the_optical_answer = False
    if (
        collecting_params["dataset_selection"]["L7"]
        or collecting_params["dataset_selection"]["L8"]
        or collecting_params["dataset_selection"]["L9"]
        or collecting_params["dataset_selection"]["MO"]
        or collecting_params["dataset_selection"]["S2"]
    ):
        the_optical_answer = True
    return the_optical_answer


# Gather collections and reduce function
def gather_collections_and_reduce(gather_collections_args):
    verbose = gather_collections_args["verbose"]

    if verbose:
        print("VERBOSE!")
        print(
            "Arguments inside afn_gather_collections_and_reduce",
            gather_collections_args,
        )

    dataset_selection = gather_collections_args["dataset_selection"]
    default_study_area = gather_collections_args["default_study_area"]
    day_step_size = gather_collections_args["day_step_size"]
    band_name_reduction = gather_collections_args["band_name_reduction"]
    which_reduction = gather_collections_args["which_reduction"]

    which_years = []
    group_start_doy = 365
    group_end_doy = 1

    if dataset_selection["L5"]:
        sensor_input_l5 = "LANDSAT/LT05/C02/T1_L2"
        cloud_cover_name_input_l5 = "CLOUD_COVER"
        cloud_cover_threshold_l5 = gather_collections_args["L5dictionary"][
            "cloud_cover_threshold"
        ]
        which_years.extend(gather_collections_args["L5dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["L5dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["L5dictionary"]["last_doy"])
        )

    if dataset_selection["L7"]:
        sensor_input_l7 = "LANDSAT/LE07/C02/T1_L2"
        cloud_cover_name_input_l7 = "CLOUD_COVER"
        cloud_cover_threshold_l7 = gather_collections_args["L7dictionary"][
            "cloud_cover_threshold"
        ]
        which_years.extend(gather_collections_args["L7dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["L7dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["L7dictionary"]["last_doy"])
        )

    if dataset_selection["L8"]:
        sensor_input_l8 = "LANDSAT/LC08/C02/T1_L2"
        cloud_cover_name_input_l8 = "CLOUD_COVER"
        cloud_cover_threshold_l8 = gather_collections_args["L8dictionary"][
            "cloud_cover_threshold"
        ]
        which_years.extend(gather_collections_args["L8dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["L8dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["L8dictionary"]["last_doy"])
        )

    if dataset_selection["L9"]:
        sensor_input_l9 = "LANDSAT/LC09/C02/T1_L2"
        cloud_cover_name_input_l9 = "CLOUD_COVER"
        cloud_cover_threshold_l9 = gather_collections_args["L9dictionary"][
            "cloud_cover_threshold"
        ]
        which_years.extend(gather_collections_args["L9dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["L9dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["L9dictionary"]["last_doy"])
        )

    if dataset_selection["MO"]:
        sensor_input_mo = "MODIS/006/MCD43A4"
        which_years.extend(gather_collections_args["MOdictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["MOdictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["MOdictionary"]["last_doy"])
        )
        modis_incorporation_style = gather_collections_args["MOdictionary"][
            "incorporation_style"
        ]

    if dataset_selection["S1"]:
        sensor_input_s1 = "COPERNICUS/S1_GRD"
        which_years.extend(gather_collections_args["S1dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["S1dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["S1dictionary"]["last_doy"])
        )

    if dataset_selection["S2"]:
        sensor_input_s2 = "COPERNICUS/S2_SR_HARMONIZED"
        cloud_cover_name_input_s2 = "CLOUD_COVERAGE_ASSESSMENT"
        cloud_cover_threshold_s2 = gather_collections_args["S2dictionary"][
            "cloud_cover_threshold"
        ]
        which_years.extend(gather_collections_args["S2dictionary"]["years_list"])
        group_start_doy = min(
            group_start_doy, int(gather_collections_args["S2dictionary"]["first_doy"])
        )
        group_end_doy = max(
            group_end_doy, int(gather_collections_args["S2dictionary"]["last_doy"])
        )

        cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        cs_plus_bands = cs_plus.first().bandNames()
        linked_s2_and_cloud_score_plus_ic = ee.ImageCollection(
            sensor_input_s2
        ).linkCollection(cs_plus, cs_plus_bands)

    which_years = sorted(set(which_years))

    if verbose:
        print("Verbose which_years before bubble_sort", which_years)
        print("Verbose Minimum Start DOY among collections", group_start_doy)
        print("Maximum End DOY among collections", group_end_doy)

    def afn_get_germane_values_for_a_given_period_all_sensors(
        get_germane_values_parameters,
    ):
        which_years = get_germane_values_parameters["which_years"]
        group_start_doy = get_germane_values_parameters["group_start_doy"]
        group_end_doy = get_germane_values_parameters["group_end_doy"]
        day_step_size = get_germane_values_parameters["day_step_size"]
        default_study_area = get_germane_values_parameters["default_study_area"]
        which_reduction = get_germane_values_parameters["which_reduction"]
        band_name_reduction = get_germane_values_parameters["band_name_reduction"]

        # start_month = int(group_start_doy / 30) + 1
        # start_day = group_start_doy % 30 + 1
        # end_month = int(group_end_doy / 30) + 1
        # end_day = group_end_doy % 30 + 1

        start_month = doy_to_month(group_start_doy, 1970)
        start_day = doy_to_day(group_start_doy, 1970)
        end_month = doy_to_month(group_end_doy, 1970)
        end_day = doy_to_day(group_end_doy, 1970)

        day_step_size_millis = day_step_size * 24 * 60 * 60 * 1000

        def afn_nested_year(the_year):
            the_year = ee.Number(the_year)
            start_date_millis = ee.Date.fromYMD(
                the_year, start_month, start_day
            ).millis()
            end_date_millis = ee.Date.fromYMD(the_year, end_month, end_day).millis()
            if start_month > end_month:
                end_date_millis = ee.Date.fromYMD(
                    the_year + 1, end_month, end_day
                ).millis()

            list_of_millis = ee.List.sequence(
                start_date_millis, end_date_millis, day_step_size_millis
            )

            def afn_nested_day(bin_start_millis):
                start = ee.Date(bin_start_millis)
                end = ee.Date(bin_start_millis).advance(day_step_size, "day")
                multi_sensor_time_slice = ee.ImageCollection([])

                if dataset_selection["DW"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_dw)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                    )
                    one_time_slice = one_time_slice.merge(ee.ImageCollection([]))
                    if which_reduction == "mode":
                        one_time_slice_reduction = one_time_slice.select("label").mode()
                        multi_sensor_time_slice = multi_sensor_time_slice.merge(
                            one_time_slice_reduction
                        )

                if dataset_selection["L5"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_l5)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                        .filter(
                            ee.Filter.lt(
                                cloud_cover_name_input_l5, cloud_cover_threshold_l5
                            )
                        )
                    )
                    one_time_slice_cloud_masked = one_time_slice.map(
                        cloud_mask_ic_l5_and_l7
                    )
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B4", "SR_B7"])
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B4", "SR_B3"])
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice_cloud_masked.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.select("SR_B5").rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["L7"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_l7)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                        .filter(
                            ee.Filter.lt(
                                cloud_cover_name_input_l7, cloud_cover_threshold_l7
                            )
                        )
                    )
                    one_time_slice_cloud_masked = one_time_slice.map(
                        cloud_mask_ic_l5_and_l7
                    )
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B4", "SR_B7"])
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B4", "SR_B3"])
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice_cloud_masked.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.select("SR_B5").rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["L8"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_l8)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                        .filter(
                            ee.Filter.lt(
                                cloud_cover_name_input_l8, cloud_cover_threshold_l8
                            )
                        )
                    )
                    one_time_slice_cloud_masked = one_time_slice.map(
                        mask_sr_clouds_l8_and_l9
                    ).map(apply_scale_factors_l89)
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B5", "SR_B7"])
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B5", "SR_B4"])
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice_cloud_masked.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.select("SR_B6").rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["L9"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_l9)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                        .filter(
                            ee.Filter.lt(
                                cloud_cover_name_input_l9, cloud_cover_threshold_l9
                            )
                        )
                    )
                    one_time_slice_cloud_masked = one_time_slice.map(
                        mask_sr_clouds_l8_and_l9
                    ).map(apply_scale_factors_l89)
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B5", "SR_B7"])
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(["SR_B5", "SR_B4"])
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice_cloud_masked.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.select("SR_B6").rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["MO"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_mo)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                    )
                    one_time_slice_cloud_masked = one_time_slice
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(
                                ["Nadir_Reflectance_Band2", "Nadir_Reflectance_Band7"]
                            )
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.normalizedDifference(
                                ["Nadir_Reflectance_Band2", "Nadir_Reflectance_Band1"]
                            )
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice_cloud_masked.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice_cloud_masked.map(
                            lambda img: img.select("Nadir_Reflectance_Band6").rename(
                                band_name_reduction
                            )
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["S1"]:
                    one_time_slice = (
                        ee.ImageCollection(sensor_input_s1)
                        .filterBounds(default_study_area)
                        .filterDate(start, end)
                    )
                    sar_to_track = gather_collections_args["S1dictionary"][
                        "sar_value_to_track"
                    ]
                    if sar_to_track == "HH":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.select(["HH"]).rename(band_name_reduction)
                        )
                    if sar_to_track == "HV":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.select(["HV"]).rename(band_name_reduction)
                        )
                    if sar_to_track == "VH":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.select(["VH"]).rename(band_name_reduction)
                        )
                    if sar_to_track == "VV":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.select(["VV"]).rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if dataset_selection["S2"]:
                    if the_year >= 2021:
                        one_time_slice = (
                            linked_s2_and_cloud_score_plus_ic.filterBounds(
                                default_study_area
                            )
                            .filterDate(start, end)
                            .map(mask_low_qa)
                            .map(apply_scale_factors_s2)
                        )
                    elif the_year >= 2019:
                        s2_cloudless_params = {
                            "s2cloudless": gather_collections_args["S2dictionary"][
                                "s2cloudless"
                            ],
                            "default_study_area": default_study_area,
                            "start_millis": start,
                            "end_millis": end,
                        }
                        s2_cloudless_params["s2cloudless"][
                            "cloud_cover_threshold_s2"
                        ] = cloud_cover_threshold_s2
                        one_time_slice = (
                            ee.ImageCollection(sensor_input_s2)
                            .filterBounds(default_study_area)
                            .filterDate(start, end)
                            .map(apply_scale_factors_s2)
                        )
                    else:
                        one_time_slice = (
                            ee.ImageCollection(sensor_input_s2)
                            .filterBounds(default_study_area)
                            .filterDate(start, end)
                            .filter(
                                ee.Filter.lt(
                                    cloud_cover_name_input_s2, cloud_cover_threshold_s2
                                )
                            )
                            .map(cloud_mask_ic_older_s2)
                            .map(apply_scale_factors_s2)
                        )
                    if which_reduction == "NBR":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.normalizedDifference(["B8", "B12"])
                        )
                    if which_reduction == "NDVI":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.normalizedDifference(["B8", "B4"])
                        )
                    if which_reduction in ["count", "binary"]:
                        one_time_slice_reduction = (
                            one_time_slice.select(0)
                            .count()
                            .toInt()
                            .rename([band_name_reduction])
                        )
                    if which_reduction == "SWIR":
                        one_time_slice_reduction = one_time_slice.map(
                            lambda img: img.select("B11")
                            .divide(1e4)
                            .rename(band_name_reduction)
                        )
                    multi_sensor_time_slice = multi_sensor_time_slice.merge(
                        one_time_slice_reduction
                    )

                if which_reduction == "mode":
                    daily_answer = multi_sensor_time_slice.mode().rename(
                        [band_name_reduction]
                    )
                else:
                    daily_answer = multi_sensor_time_slice.median().rename(
                        [band_name_reduction]
                    )

                daily_answer = daily_answer.set("nominal_date", end)
                daily_answer = daily_answer.set(
                    "nominal_doy", end.getRelative("day", "year")
                )
                daily_answer = daily_answer.set(
                    "frac_doy", ee.Number(end.getRelative("day", "year")).divide(365)
                )
                daily_answer = daily_answer.set(
                    "float_date", end.difference(ee.Date("2015-01-01"), "year")
                )
                daily_answer = daily_answer.set("the_sensor", "mixed")
                daily_answer = daily_answer.set("millis", end.millis())
                daily_answer = daily_answer.set("system:time_start", end.millis())
                return daily_answer

            yearly_answer = list_of_millis.map(afn_nested_day).flatten()
            return yearly_answer

        multi_year_answer = ee.List(which_years).map(afn_nested_year)
        # multi_year_answer = map(afn_nested_year, which_years)
        the_list = ee.List(multi_year_answer).flatten()
        the_collection = ee.ImageCollection.fromImages(the_list)
        return the_collection

    get_germane_values_parameters = {
        "which_years": which_years,
        "group_start_doy": group_start_doy,
        "group_end_doy": group_end_doy,
        "day_step_size": day_step_size,
        "default_study_area": default_study_area,
        "which_reduction": which_reduction,
        "band_name_reduction": band_name_reduction,
    }

    gathered_reduced_multi_sensor_values = (
        afn_get_germane_values_for_a_given_period_all_sensors(
            get_germane_values_parameters
        )
    )
    gathered_reduced_multi_sensor_values = ee.ImageCollection(
        gathered_reduced_multi_sensor_values
    )
    collection = ee.ImageCollection([]).merge(gathered_reduced_multi_sensor_values)
    return collection


def reduce_collection_to_points_and_write_to_file(collection, points, output_file_path):
    measurements = pd.DataFrame(columns=["longitude", "latitude", "swir", "date"])
    for point_index, point in enumerate(points):
        point_geometry = ee.Geometry.Point(point)
        collection_for_point = collection.filterBounds(point_geometry)

        def process_image(image):
            img = image

            def sample_and_copy(feature):
                return feature.copyProperties(
                    img, img.propertyNames().remove(ee.String("nominalDate"))
                )

            sampled = img.sample(
                region=point_geometry,
                scale=10,
                dropNulls=False,
                geometries=True,
            ).map(sample_and_copy)

            return sampled

        features = (
            collection_for_point.map(process_image).flatten().getInfo()["features"]
        )

        data = pd.DataFrame(
            [
                [
                    int(point_index),
                    feature["geometry"]["coordinates"][0],
                    feature["geometry"]["coordinates"][1],
                    (
                        feature["properties"]["swir"]
                        if "swir" in feature["properties"]
                        else 0
                    ),
                    feature["properties"]["millis"],
                ]
                for feature in features
            ],
            columns=["point", "longitude", "latitude", "swir", "date"],
        )

        measurements = (
            measurements.copy()
            if data.empty
            else (
                data.copy()
                if measurements.empty
                else pd.concat([measurements, data], ignore_index=True)
            )
        )

    measurements = measurements[["point", "longitude", "latitude", "swir", "date"]]
    measurements.to_csv(output_file_path, index=False, mode="w")
