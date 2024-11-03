import ee

# CONSTANTS

# Conversion factor from ms to days
MS_TO_DAYS = 86400000
# Number of days in common era until epoch 01-01-1970 (non-inclusive)
EPOCH_DAYS = 719529

# FUNCTIONS


def ms_to_days(ms):
    """Convert milliseconds since epoch (01-01-1970) to number of days."""
    return ee.Number(ms).divide(MS_TO_DAYS)


def date_to_jdays(str_date):
    """Convert Date to Julian days in common era (i.e. days since 00-00-0000)."""
    if not str_date:
        raise ValueError("Required parameter [str_date] missing")
    date = ee.Date(str_date)
    return ms_to_days(date.millis()).add(EPOCH_DAYS)


def jdays_to_ms(jdays):
    """Convert Julian day in common era to ms since 1970-01-01."""
    days_since_epoch = ee.Number(jdays).subtract(EPOCH_DAYS)
    return days_since_epoch.multiply(MS_TO_DAYS)


def jdays_to_date(jdays):
    """Convert Julian day in common era to ee.Date."""
    return ee.Date(jdays_to_ms(jdays))


def ms_to_jdays(ms):
    """Convert ms since 1970-01-01 to Julian day in common era."""
    return ms_to_days(ms).add(EPOCH_DAYS)


def ms_to_frac(ms):
    """Convert ms since 1970-01-01 to fractional year."""
    year = ee.Date(ms).get("year")
    frac = ee.Date(ms).getFraction("year")
    return year.add(frac)


def frac_to_ms(frac):
    """Convert fractional time to ms since 1970-01-01."""
    fyear = ee.Number(frac)
    year = fyear.floor()
    d = fyear.subtract(year).multiply(365.25)
    day_one = ee.Date.fromYMD(year, 1, 1)
    return day_one.advance(d, "day").millis()


def frac_to_date(frac):
    """Convert fractional time to ee.Date."""
    ms = frac_to_ms(frac)
    return ms_to_date(ms)


def ms_to_date(ms):
    """Convert ms to ee.Date."""
    return jdays_to_date(ms_to_jdays(ms))


def convert_date(options):
    """Convert between any two date formats."""
    input_format = options.get("input_format", 0)
    input_date = options.get("input_date", None)
    output_format = options.get("output_format", 0)
    if input_date is None:
        raise ValueError("Required parameter [input_date] missing")

    # First convert to millis
    if input_format == 0:
        milli = jdays_to_ms(input_date)
    elif input_format == 1:
        milli = frac_to_ms(input_date)
    elif input_format == 2:
        milli = input_date
    elif input_format == 3:
        milli = jdays_to_ms(date_to_jdays(input_date))

    # Now convert to output format
    if output_format == 0:
        output = ms_to_jdays(milli)
    elif output_format == 1:
        output = ms_to_frac(milli)
    elif output_format == 2:
        output = milli
    elif output_format == 4:
        output = jdays_to_date(ms_to_jdays(milli))

    return output
