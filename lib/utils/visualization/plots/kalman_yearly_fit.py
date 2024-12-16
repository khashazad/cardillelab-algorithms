import pandas as pd
from pprint import pprint
import matplotlib.dates as mdates
import csv
from lib.utils.harmonic import extract_coefficients_from_array
from lib.utils.harmonic import calculate_harmonic_fit
from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    CCDC,
    DATE_LABEL,
    FRACTION_OF_YEAR_LABEL,
    HARMONIC_FLAGS_LABEL,
    RETROFITTED_TREND_LABEL,
    Harmonic,
    Kalman,
)


def calculate_ccdc_fit(data):
    ccdc_coef = lambda coef: f"{CCDC.BAND_PREFIX.value}_{coef}"

    data["CCDC"] = data.apply(
        lambda row: calculate_harmonic_fit(
            {
                Harmonic.INTERCEPT.value: row[ccdc_coef(Harmonic.INTERCEPT.value)],
                Harmonic.SLOPE.value: row[ccdc_coef(Harmonic.SLOPE.value)],
                Harmonic.COS.value: row[ccdc_coef(Harmonic.COS.value)],
                Harmonic.SIN.value: row[ccdc_coef(Harmonic.SIN.value)],
                Harmonic.COS2.value: row[ccdc_coef(Harmonic.COS2.value)],
                Harmonic.SIN2.value: row[ccdc_coef(Harmonic.SIN2.value)],
                Harmonic.COS3.value: row[ccdc_coef(Harmonic.COS3.value)],
                Harmonic.SIN3.value: row[ccdc_coef(Harmonic.SIN3.value)],
            },
            row[FRACTION_OF_YEAR_LABEL],
        ),
        axis=1,
    )

    return data


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


def kalman_yearly_fit_plot(
    axs,
    data,
    options,
    year,
):
    coefs_array = get_end_of_year_coefficients(options)[str(year)]

    harmonic_flags = options.get(HARMONIC_FLAGS_LABEL, {})

    coefs = extract_coefficients_from_array(coefs_array, harmonic_flags)

    unique_years = pd.to_datetime(data[DATE_LABEL]).dt.year.unique().tolist()

    for year in unique_years:
        axs.axvline(
            x=pd.Timestamp(year=year, month=1, day=1),
            color="gray",
            linestyle="-",
            alpha=0.1,
        )

    data["yearly_fit"] = data.apply(
        lambda row: calculate_harmonic_fit(coefs, row[FRACTION_OF_YEAR_LABEL]), axis=1
    )

    ccdc_coef = lambda coef: f"{CCDC.BAND_PREFIX.value}_{coef}"

    ccdc_filtered = data[
        (data[ccdc_coef(Harmonic.INTERCEPT.value)] != 0)
        | (data[ccdc_coef(Harmonic.SLOPE.value)] != 0)
        | (data[ccdc_coef(Harmonic.COS.value)] != 0)
        | (data[ccdc_coef(Harmonic.SIN.value)] != 0)
        | (data[ccdc_coef(Harmonic.COS2.value)] != 0)
        | (data[ccdc_coef(Harmonic.SIN2.value)] != 0)
        | (data[ccdc_coef(Harmonic.COS3.value)] != 0)
        | (data[ccdc_coef(Harmonic.SIN3.value)] != 0)
    ]

    last_index = ccdc_filtered.last_valid_index()
    last_valid_coefs = ccdc_filtered.iloc[last_index]

    condition = (
        (data[ccdc_coef(Harmonic.INTERCEPT.value)] == 0)
        & (data[ccdc_coef(Harmonic.SLOPE.value)] == 0)
        & (data[ccdc_coef(Harmonic.COS.value)] == 0)
        & (data[ccdc_coef(Harmonic.SIN.value)] == 0)
        & (data[ccdc_coef(Harmonic.COS2.value)] == 0)
        & (data[ccdc_coef(Harmonic.SIN2.value)] == 0)
        & (data[ccdc_coef(Harmonic.COS3.value)] == 0)
        & (data[ccdc_coef(Harmonic.SIN3.value)] == 0)
    )

    ccdc_coefs = [
        ccdc_coef(Harmonic.INTERCEPT.value),
        ccdc_coef(Harmonic.SLOPE.value),
        ccdc_coef(Harmonic.COS.value),
        ccdc_coef(Harmonic.SIN.value),
        ccdc_coef(Harmonic.COS2.value),
        ccdc_coef(Harmonic.SIN2.value),
        ccdc_coef(Harmonic.COS3.value),
        ccdc_coef(Harmonic.SIN3.value),
    ]

    data.loc[~condition, ccdc_coefs] = last_valid_coefs[ccdc_coefs]

    data = calculate_ccdc_fit(data)

    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    axs.plot(
        data[DATE_LABEL],
        data["CCDC"],
        label="CCDC Fit",
        linestyle="--",
        color="green",
    )

    axs.plot(
        data[DATE_LABEL],
        data["yearly_fit"],
        label=f"{year} Fit",
        linestyle="-",
        color="blue",
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

    axs.set_title(f"{options.get('title', '')} - {year}")
