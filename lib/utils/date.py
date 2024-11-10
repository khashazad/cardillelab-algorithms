import pandas as pd


def convert_to_fraction_of_year(date: pd.Timestamp):
    return (date - pd.Timestamp("2016-01-01")).total_seconds() / (365.25 * 24 * 60 * 60)
