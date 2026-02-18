"""
Load and clean raw data sources.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ili_data(filepath: str, country: str = "ISR") -> pd.DataFrame:
    """
    Load WHO FluID data and extract weekly ILI series for a given country.

    Parameters
    ----------
    filepath : str
        Path to VIW_FID_EPI.csv
    country : str
        ISO3 country code

    Returns
    -------
    pd.DataFrame with columns: week_start_date, ILI_CASE
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Convert types
    df["ILI_CASE"] = pd.to_numeric(df["ILI_CASE"], errors="coerce")
    df["ISO_WEEK"] = pd.to_numeric(df["ISO_WEEK"], errors="coerce")
    df["ISO_WEEKSTARTDATE"] = pd.to_datetime(df["ISO_WEEKSTARTDATE"], errors="coerce")

    # Filter country and aggregate across age groups
    country_df = df[df["COUNTRY_CODE"] == country].copy()
    weekly = (
        country_df
        .groupby("ISO_WEEKSTARTDATE", as_index=False)["ILI_CASE"]
        .sum()
    )
    weekly.columns = ["week_start_date", "ILI_CASE"]

    # Shift ISO Monday → Sunday (Israeli convention)
    weekly["week_start_date"] = weekly["week_start_date"] - pd.Timedelta(days=1)

    return weekly.sort_values("week_start_date").reset_index(drop=True)


def load_southern_hemisphere(filepath: str, countries: list = None) -> pd.DataFrame:
    """
    Load ILI data for Southern Hemisphere countries (leading indicators).

    Returns
    -------
    pd.DataFrame with columns: week_start_date, {COUNTRY}_ILI_CASE for each country
    """
    if countries is None:
        countries = ["AUS"]

    df = pd.read_csv(filepath, low_memory=False)
    df["ILI_CASE"] = pd.to_numeric(df["ILI_CASE"], errors="coerce")
    df["ISO_WEEKSTARTDATE"] = pd.to_datetime(df["ISO_WEEKSTARTDATE"], errors="coerce")

    result = None

    for code in countries:
        country_df = df[df["COUNTRY_CODE"] == code].copy()
        weekly = (
            country_df
            .groupby("ISO_WEEKSTARTDATE", as_index=False)["ILI_CASE"]
            .sum()
        )
        weekly.columns = ["week_start_date", f"{code}_ILI_CASE"]
        weekly["week_start_date"] = weekly["week_start_date"] - pd.Timedelta(days=1)

        if result is None:
            result = weekly
        else:
            result = result.merge(weekly, on="week_start_date", how="outer")

    return result.sort_values("week_start_date").reset_index(drop=True)


def load_temperature(filepath: str) -> pd.DataFrame:
    """
    Load daily temperature data and aggregate to weekly.

    Returns
    -------
    pd.DataFrame with columns: week_start_date, tmean_c
    """
    temp = pd.read_csv(filepath)

    # Identify date and temperature columns (adapt to your CSV structure)
    # Assumes columns: date (or similar), tmax, tmin
    date_col = [c for c in temp.columns if "date" in c.lower() or "תאריך" in c][0]
    temp["date"] = pd.to_datetime(temp[date_col], dayfirst=True, errors="coerce")

    # Find max/min temperature columns (English or Hebrew variants)
    tmax_col = [c for c in temp.columns if "max" in c.lower() or "עליונה" in c or "מקסימום" in c]
    tmin_col = [c for c in temp.columns if "min" in c.lower() or "תחתונה" in c or "מינימום" in c]

    if tmax_col and tmin_col:
        temp["tmax"] = pd.to_numeric(temp[tmax_col[0]], errors="coerce")
        temp["tmin"] = pd.to_numeric(temp[tmin_col[0]], errors="coerce")
        temp["tmean"] = (temp["tmax"] + temp["tmin"]) / 2
    else:
        raise ValueError("Cannot identify temperature columns in file")

    temp = temp.dropna(subset=["date", "tmean"])
    temp = temp.set_index("date")

    # Resample to weekly (Sunday start)
    weekly_temp = temp["tmean"].resample("W-SUN").mean().reset_index()
    weekly_temp.columns = ["week_start_date", "tmean_c"]

    return weekly_temp


def build_continuous_weekly_grid(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create a continuous Sunday-based weekly grid.

    Returns
    -------
    pd.DataFrame with columns: week_start_date, year, week
    """
    grid = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
    df = pd.DataFrame({"week_start_date": grid})
    df["year"] = df["week_start_date"].dt.isocalendar().year.astype(int)
    df["week"] = df["week_start_date"].dt.isocalendar().week.astype(int)
    return df
