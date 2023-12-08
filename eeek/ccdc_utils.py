"""Python port of CCDC Utilities.

Orginial JavaScript version, written by Paulo Arevalo, can be found on Google
Earth Engine at "users/parevalo_bu/gee-ccdc-tools"
"""
import ee


HARMONIC_TAGS = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]


def filter_coefs(ccdc_results, date, band, coef, segment_names, behavior):
    if behavior not in ["normal", "after", "before"]:
        raise NotImplementedError(
            f"behavior must be 'normal', 'after', or 'before'; got {behavior}"
        )

    start_bands = ccdc_results.select(".*_tStart").rename(segment_names)
    end_bands = ccdc_results.select(".*_tEnd").rename(segment_names)

    sel_str = ".*" + band + "_.*" + coef
    coef_bands = ccdc_results.select(sel_str)

    normal_start = start_bands.lte(date)
    normal_end = end_bands.gte(date)

    segment_match = ee.Algorithms.If(
        behavior == "normal",
        normal_start.And(normal_end),
        ee.Algorithms.If(
            behavior == "after",
            end_bands.gt(date),
            ee.Algorithms.If(
                behavior == "before",
                start_bands.selfMask().lt(date).selfMask(),
                None,  # NotImplementedError should already have been thrown
            ),
        ),
    )

    return coef_bands.updateMask(segment_match).reduce(ee.Reducer.firstNonNull())


def get_coef(ccdc_results, date, band_list, coef, segment_names, behavior):
    def inner(band):
        band_coef = filter_coefs(
            ccdc_results, date, band, coef, segment_names, behavior
        )
        return band_coef.rename(band + "_" + coef)

    return ee.Image([inner(b) for b in band_list])


def normalize_intercept(intercepts, start, end, slope):
    middle_date = ee.Image(start).add(ee.Image(end)).divide(2)
    slope_coef = ee.Image(slope).multiply(middle_date)
    return ee.Image(intercepts).add(slope_coef)


def apply_norm(band_coefs, segment_start, segment_end):
    intercepts = band_coefs.select(".*INTP")
    slopes = band_coefs.select(".*SLP")
    normalized = normalize_intercept(intercepts, segment_start, segment_end, slopes)
    return band_coefs.addBands(normalized, overwrite=True)


def get_multi_coefs(
    ccdc_results,
    date,
    band_list,
    coef_list=None,
    cond=True,
    segment_names=None,
    behavior="after",
):
    if coef_list is None:
        coef_list = HARMONIC_TAGS

    if segment_names is None:
        segment_names = build_segment_tag(10)  # default to 10 tags...?

    def inner(coef):
        return get_coef(ccdc_results, date, band_list, coef, segment_names, behavior)

    coefs = ee.Image([inner(c) for c in coef_list])

    seg_start = filter_coefs(ccdc_results, date, "", "tStart", segment_names, behavior)
    seg_end = filter_coefs(ccdc_results, date, "", "tEnd", segment_names, behavior)
    norm_coefs = apply_norm(coefs, seg_start, seg_end)

    return ee.Image(ee.Algorithms.If(cond, norm_coefs, coefs))


def build_band_tag(tag, band_list):
    bands = ee.List(band_list)
    return bands.map(lambda s: ee.String(s).cat("_" + tag))


def build_segment_tag(n_segments):
    return ee.List(["S" + str(x) for x in range(1, n_segments + 1)])


def extract_band(fit, n_segments, band_list, in_prefix, out_prefix):
    segment_tag = build_segment_tag(n_segments)
    zeros = ee.Image(ee.Array(ee.List.repeat(0, n_segments)))

    def retrieve(band):
        mag_img = (
            fit.select(band + in_prefix)
            .arrayCat(zeros, 0)
            .float()
            .arraySlice(0, 0, n_segments)
        )
        tags = segment_tag.map(
            lambda x: ee.String(x).cat("_").cat(band).cat(out_prefix)
        )
        return mag_img.arrayFlatten([tags])

    return ee.Image([retrieve(b) for b in band_list])


def build_start_end_break_prob(fit, n_segments, tag):
    segment_tag = build_segment_tag(n_segments).map(
        lambda s: ee.String(s).cat("_" + tag)
    )

    zeros = ee.Array(0).repeat(0, n_segments)
    mag_img = fit.select(tag).arrayCat(zeros, 0).float().arraySlice(0, 0, n_segments)

    return mag_img.arrayFlatten([segment_tag])


def build_coefs(fit, n_segments, band_list):
    segment_tag = build_segment_tag(n_segments)

    zeros = ee.Image(ee.Array([ee.List.repeat(0, len(HARMONIC_TAGS))])).arrayRepeat(
        0, n_segments
    )

    def retrieve_coefs(band):
        coef_img = (
            fit.select(band + "_coefs")
            .arrayCat(zeros, 0)
            .float()
            .arraySlice(0, 0, n_segments)
        )
        tags = segment_tag.map(lambda x: ee.String(x).cat("_").cat(band).cat("_coef"))
        return coef_img.arrayFlatten([tags, HARMONIC_TAGS])

    return ee.Image([retrieve_coefs(b) for b in band_list])


def build_ccd_image(fit, n_segments, band_list):
    magnitude = extract_band(fit, n_segments, band_list, "_magnitude", "_MAG")
    rmse = extract_band(fit, n_segments, band_list, "_rmse", "_RMSE")

    coef = build_coefs(fit, n_segments, band_list)
    t_start = build_start_end_break_prob(fit, n_segments, "tStart")
    t_end = build_start_end_break_prob(fit, n_segments, "tEnd")
    t_break = build_start_end_break_prob(fit, n_segments, "tBreak")
    probs = build_start_end_break_prob(fit, n_segments, "changeProb")
    n_obs = build_start_end_break_prob(fit, n_segments, "numObs")

    return ee.Image.cat(coef, rmse, magnitude, t_start, t_end, t_break, probs, n_obs)
