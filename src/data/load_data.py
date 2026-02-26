"""
Load and clean raw data sources.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml
from epiweeks import Week

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate that all required config keys and files exist."""
    required_keys = [
        "paths.raw_ili", "paths.raw_temperature", "paths.raw_holidays",
        "paths.processed", "paths.models_dir", "paths.figures_dir",
        "data.country", "data.start_date", "data.end_date",
        "sarimax.order", "sarimax.seasonal_order", "sarimax.exog_features",
        "split.split_date",
    ]
    for dotted_key in required_keys:
        parts = dotted_key.split(".")
        node = config
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Missing required config key: {dotted_key}")
            node = node[part]

    # Check that raw data files exist
    for key in ["raw_ili", "raw_temperature", "raw_holidays"]:
        filepath = config["paths"][key]
        if not Path(filepath).exists():
            logger.warning(f"Data file not found: {filepath} (key: paths.{key})")

    # Validate date range
    start = pd.Timestamp(config["data"]["start_date"])
    end = pd.Timestamp(config["data"]["end_date"])
    if start >= end:
        raise ValueError(f"start_date ({start}) must be before end_date ({end})")

    split = pd.Timestamp(config["split"]["split_date"])
    if not (start < split < end):
        raise ValueError(f"split_date ({split}) must be between start_date and end_date")

    logger.info("Config validation passed")


# ── Shared ILI loader ─────────────────────────────────────────────────
def _load_country_ili(filepath: str, country_code: str) -> pd.DataFrame:
    """
    Shared loader for any country's weekly ILI data from WHO FluID.

    Returns
    -------
    pd.DataFrame with columns: week_start_date, ILI_CASE
    """
    df = pd.read_csv(filepath, low_memory=False)
    df["ILI_CASE"] = pd.to_numeric(df["ILI_CASE"], errors="coerce")
    df["MMWR_WEEKSTARTDATE"] = pd.to_datetime(df["MMWR_WEEKSTARTDATE"], errors="coerce")

    country_df = df[df["COUNTRY_CODE"] == country_code].copy()
    if country_df.empty:
        logger.warning(f"No data found for country code: {country_code}")

    weekly = (
        country_df
        .groupby("MMWR_WEEKSTARTDATE", as_index=False)["ILI_CASE"]
        .sum()
    )
    weekly.columns = ["week_start_date", "ILI_CASE"]

    return weekly.sort_values("week_start_date").reset_index(drop=True)


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
    return _load_country_ili(filepath, country)


def load_southern_hemisphere(filepath: str, countries: list = None) -> pd.DataFrame:
    """
    Load ILI data for Southern Hemisphere countries (leading indicators).

    Returns
    -------
    pd.DataFrame with columns: week_start_date, {COUNTRY}_ILI_CASE for each country
    """
    if countries is None:
        countries = ["AUS"]

    result = None

    for code in countries:
        weekly = _load_country_ili(filepath, code)
        weekly = weekly.rename(columns={"ILI_CASE": f"{code}_ILI_CASE"})

        if result is None:
            result = weekly
        else:
            result = result.merge(weekly, on="week_start_date", how="outer")

    if result is None:
        raise ValueError("No Southern Hemisphere countries specified")

    return result.sort_values("week_start_date").reset_index(drop=True)


def load_temperature(filepath: str) -> pd.DataFrame:
    """
    Load daily temperature data and aggregate to weekly.

    Returns
    -------
    pd.DataFrame with columns: week_start_date, tmean_c
    """
    temp = pd.read_csv(filepath)

    # Identify date column — with safe fallback
    date_candidates = [c for c in temp.columns if "date" in c.lower() or "תאריך" in c]
    if not date_candidates:
        raise ValueError(
            f"No date column found in temperature file. Available columns: {list(temp.columns)}"
        )
    date_col = date_candidates[0]
    temp["date"] = pd.to_datetime(temp[date_col], dayfirst=True, errors="coerce")

    # Find max/min temperature columns (English or Hebrew variants)
    tmax_candidates = [c for c in temp.columns if "max" in c.lower() or "עליונה" in c or "מקסימום" in c]
    tmin_candidates = [c for c in temp.columns if "min" in c.lower() or "תחתונה" in c or "מינימום" in c]

    if not tmax_candidates or not tmin_candidates:
        raise ValueError(
            f"Cannot identify temperature columns. "
            f"Found tmax candidates: {tmax_candidates}, tmin candidates: {tmin_candidates}. "
            f"Available columns: {list(temp.columns)}"
        )

    temp["tmax"] = pd.to_numeric(temp[tmax_candidates[0]], errors="coerce")
    temp["tmin"] = pd.to_numeric(temp[tmin_candidates[0]], errors="coerce")
    temp["tmean"] = (temp["tmax"] + temp["tmin"]) / 2

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
    epi = df["week_start_date"].apply(lambda d: Week.fromdate(d))
    df["year"] = epi.apply(lambda w: w.year).astype(int)
    df["week"] = epi.apply(lambda w: w.week).astype(int)
    return df
