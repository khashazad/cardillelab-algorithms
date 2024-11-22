import pandas as pd
from pprint import pprint
import matplotlib.dates as mdates
import csv
from lib.utils.harmonic import extract_coefficients_from_array, parse_harmonic_params

from lib.utils.harmonic import calculate_harmonic_estimate
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE_LABEL,
    ESTIMATE_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_FLAGS_LABEL,
    TIMESTAMP_LABEL,
    Kalman,
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

    return pd.DataFrame(
        estimates, columns=[FRACTION_OF_YEAR_LABEL, Kalman.RETROFITTED.value]
    )


def kalman_retrofitted_plot(
    axs,
    data,
    options,
):

    eoy_state_coefs = get_end_of_year_coefficients(options)
    eoy_state_df = get_retrofitted_trend(
        eoy_state_coefs,
        data[FRACTION_OF_YEAR_LABEL],
        options.get(HARMONIC_FLAGS_LABEL, {}),
    )

    data = data.merge(eoy_state_df, on=FRACTION_OF_YEAR_LABEL, how="inner")

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
        data[Kalman.RETROFITTED.value],
        label="Retrofitted",
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
