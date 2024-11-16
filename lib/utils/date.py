import pandas as pd

FRACTION_OF_YEAR = 365.25 * 24 * 60 * 60


def timestamp_to_frac_of_year(timestamp):

    return (
        pd.to_datetime(timestamp, unit="ms") - pd.Timestamp("2015-01-01")
    ).dt.total_seconds() / FRACTION_OF_YEAR
