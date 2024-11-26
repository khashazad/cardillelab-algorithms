import pandas as pd
import matplotlib.dates as mdates
import csv
from lib.utils.harmonic import extract_coefficients_from_array, parse_harmonic_params

from lib.utils.harmonic import calculate_harmonic_fit
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE_LABEL,
    ESTIMATE_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_TREND_LABEL,
    Harmonic,
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


def get_harmonic_trend_estimates(harmonic_trend_coefs, frac_of_year, harmonic_flags):
    harmonic_params, _ = parse_harmonic_params(harmonic_flags)

    harmonic_trend_coefs_by_year = []

    for frac_year in list(frac_of_year):
        year = int(frac_year)
        coefs_array = harmonic_trend_coefs.get(str(year), [])

        coefs = extract_coefficients_from_array(coefs_array, harmonic_flags)

        harmonic_trend_coefs_by_year.append((frac_year, coefs))

    estimates = [
        [
            frac_year,
            calculate_harmonic_fit(coefs, frac_year),
        ]
        for frac_year, coefs in harmonic_trend_coefs_by_year
    ]

    return pd.DataFrame(estimates, columns=[FRACTION_OF_YEAR_LABEL, Harmonic.FIT.value])


def kalman_estimate_vs_harmonic_trend_plot(
    axs,
    data,
    options,
):
    harmonic_trend_coefs = get_harmonic_trend_coefficients(options)
    harmonic_fit_df = get_harmonic_trend_estimates(
        harmonic_trend_coefs,
        data[FRACTION_OF_YEAR_LABEL],
        options.get("harmonic_flags", {}),
    )

    data = data.merge(harmonic_fit_df, on=FRACTION_OF_YEAR_LABEL, how="inner")

    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_LABEL],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE_LABEL],
        data[Harmonic.FIT.value],
        label="Harmonic Trend",
        linestyle="--",
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
