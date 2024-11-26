import pandas as pd
from pprint import pprint
import matplotlib.dates as mdates
import csv
from lib.utils.harmonic import extract_coefficients_from_array
from lib.utils.harmonic import calculate_harmonic_estimate
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    HARMONIC_TREND_LABEL,
    DATE_LABEL,
    ESTIMATE_LABEL,
    FORWARD_TREND_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_FLAGS_LABEL,
    RETROFITTED_TREND_LABEL,
    Kalman,
)


def get_harmonic_trend_coefficients(options):
    harmonic_trend = options.get(HARMONIC_TREND_LABEL, None)

    assert (
        harmonic_trend is not None
    ), "harmonic trend coefficients are required for generating kalman estimate vs harmonic trend plot"

    coef_dic = dict()

    with open(harmonic_trend, "r") as file:
        reader = csv.reader(file)

        for line in reader:
            coef_dic[line[0]] = line[1:]

    return coef_dic


def get_forward_trend(data, options, eoy_state_coefs, frac_of_year):
    harmonic_trend = options.get(HARMONIC_TREND_LABEL, {})
    harmonic_flags = options.get(HARMONIC_FLAGS_LABEL, {})

    with open(harmonic_trend, "r") as file:
        reader = csv.reader(file)

        first_year_coefs = extract_coefficients_from_array(
            next(reader)[1:], harmonic_flags
        )

    first_year = pd.to_datetime(data[DATE_LABEL]).iloc[0].year

    forward_trends = []

    for frac_year in frac_of_year:
        year = int(frac_year)

        if year == first_year:
            forward_trends.append(
                [frac_year, calculate_harmonic_estimate(first_year_coefs, frac_year)]
            )
        else:
            forward_trends.append(
                [
                    frac_year,
                    calculate_harmonic_estimate(
                        extract_coefficients_from_array(
                            eoy_state_coefs.get(str(year - 1), []), harmonic_flags
                        ),
                        frac_year,
                    ),
                ]
            )

    return pd.DataFrame(
        forward_trends, columns=[FRACTION_OF_YEAR_LABEL, FORWARD_TREND_LABEL]
    )


def get_end_of_year_coefficients(options):
    eoy_state = options.get(Kalman.EOY_STATE.value, None)

    assert (
        eoy_state is not None
    ), "end of year state is required for generating kalman retrofitted plot"

    coef_dic = dict()

    with open(eoy_state, "r") as file:
        reader = csv.reader(file)

        next(reader)

        for line in reader:
            values = line[1:]

            # remove the covariance values
            coefs = values[: len(values) // 2]

            coef_dic[line[0]] = coefs

    return coef_dic


def get_retrofitted_trend(eoy_state_coefs, frac_of_year, harmonic_flags):
    eoy_coefs_by_year = []

    for frac_year in list(frac_of_year):
        year = int(frac_year)
        coefs_array = eoy_state_coefs.get(str(year), [])

        coefs = extract_coefficients_from_array(coefs_array, harmonic_flags)

        eoy_coefs_by_year.append((frac_year, coefs))

    estimates = [
        [
            frac_year,
            calculate_harmonic_estimate(coefs, frac_year),
        ]
        for frac_year, coefs in eoy_coefs_by_year
    ]

    df = pd.DataFrame(
        estimates, columns=[FRACTION_OF_YEAR_LABEL, RETROFITTED_TREND_LABEL]
    )

    return df


def kalman_retrofitted_plot(
    axs,
    data,
    options,
):
    harmonic_flags = options.get(HARMONIC_FLAGS_LABEL, {})

    eoy_state_coefs = get_end_of_year_coefficients(options)
    eoy_state_df = get_retrofitted_trend(
        eoy_state_coefs,
        data[FRACTION_OF_YEAR_LABEL],
        harmonic_flags,
    )

    data = data.merge(eoy_state_df, on=FRACTION_OF_YEAR_LABEL, how="inner")

    if options.get(FORWARD_TREND_LABEL, False):
        forward_trend_df = get_forward_trend(
            data, options, eoy_state_coefs, data[FRACTION_OF_YEAR_LABEL]
        )

        data = data.merge(forward_trend_df, on=FRACTION_OF_YEAR_LABEL, how="inner")

    unique_years = pd.to_datetime(data[DATE_LABEL]).dt.year.unique().tolist()

    for year in unique_years:
        axs.axvline(
            x=pd.Timestamp(year=year, month=1, day=1),
            color="gray",
            linestyle="-",
            alpha=0.1,
        )

    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_LABEL],
        label="Kalman Fit",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE_LABEL],
        data[RETROFITTED_TREND_LABEL],
        label="Retrofitted",
        linestyle="-.",
        color="orange",
    )

    if options.get(FORWARD_TREND_LABEL, False):
        axs.plot(
            data[DATE_LABEL],
            data[FORWARD_TREND_LABEL],
            label="Forward Trend",
            linestyle="-.",
            color="green",
        )

    axs.scatter(
        data[(data[Kalman.Z.value] != 0)][DATE_LABEL],
        data[(data[Kalman.Z.value] != 0)][Kalman.Z.value],
        label="Observed",
        s=13,
        color="red",
    )

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", ""))
