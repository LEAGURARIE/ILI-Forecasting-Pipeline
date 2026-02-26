"""
Feature engineering pipeline — assembles all features into weekly_df.
"""
import logging

import pandas as pd
import numpy as np
from src.data.load_data import (
    load_ili_data, load_southern_hemisphere,
    load_temperature, build_continuous_weekly_grid
)
from src.features.holidays import build_holidays_df
from src.features.school_calendar import add_school_features
from src.features.temperature import add_temperature_features
from src.features.southern_hemisphere import add_southern_hemisphere_features

logger = logging.getLogger(__name__)


def build_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Returns
    -------
    weekly_df : pd.DataFrame
        Complete feature matrix
    holidays_df : pd.DataFrame
        Holiday DataFrame for Prophet (separate format)
    """
    paths = config["paths"]
    data_cfg = config["data"]
    feat_cfg = config["features"]

    # 1. Load Israel ILI
    isr_weekly = load_ili_data(paths["raw_ili"], data_cfg["country"])
    logger.info(f"Loaded {len(isr_weekly)} weeks of Israel ILI data")

    # 2. Build continuous weekly grid
    grid = build_continuous_weekly_grid(data_cfg["start_date"], data_cfg["end_date"])

    # 3. Merge and interpolate
    weekly_df = grid.merge(isr_weekly, on="week_start_date", how="left")
    n_missing = weekly_df["ILI_CASE"].isna().sum()
    if n_missing > 0:
        logger.warning(f"Interpolating {n_missing} missing ILI values ({n_missing/len(weekly_df)*100:.1f}%)")
    weekly_df["ILI_CASE"] = weekly_df["ILI_CASE"].interpolate(method="linear")

    # 4. Log transform
    weekly_df["log_ILI"] = np.log1p(weekly_df["ILI_CASE"])

    # 5. Southern Hemisphere regressors
    weekly_df = add_southern_hemisphere_features(
        weekly_df, paths["raw_ili"],
        data_cfg["southern_hemisphere"]["countries"],
        data_cfg["southern_hemisphere"]["lags"]
    )

    # 6. Temperature
    weekly_df = add_temperature_features(
        weekly_df, paths["raw_temperature"],
        feat_cfg["temperature"]["lags"]
    )

    # 7. COVID flags
    covid_start = pd.Timestamp(feat_cfg["covid"]["start"])
    covid_end = pd.Timestamp(feat_cfg["covid"]["end"])
    weekly_df["is_covid"] = (
        (weekly_df["week_start_date"] >= covid_start) &
        (weekly_df["week_start_date"] <= covid_end)
    ).astype(int)
    weekly_df["post_covid"] = (
        weekly_df["week_start_date"] > covid_end
    ).astype(int)

    # 8. School calendar
    weekly_df = add_school_features(weekly_df, feat_cfg["school"])

    # 9. Fill edge NaNs — only for specific columns that are safe to fill
    #    (temperature and southern hemisphere features have known edge NaN
    #     from lagging; bfill is acceptable for the first 1-2 rows)
    safe_fill_cols = _get_safe_fill_columns(weekly_df, feat_cfg)
    for col in safe_fill_cols:
        if col in weekly_df.columns:
            weekly_df[col] = weekly_df[col].bfill().ffill()

    # Log remaining NaNs
    remaining_nans = weekly_df.isna().sum()
    remaining_nans = remaining_nans[remaining_nans > 0]
    if len(remaining_nans) > 0:
        logger.warning(f"Remaining NaNs after fill:\n{remaining_nans}")

    # 10. Holidays (separate DF for Prophet)
    holidays_df = build_holidays_df(
        paths["raw_holidays"],
        feat_cfg["holidays"],
        data_cfg["start_date"],
        data_cfg["end_date"]
    )

    logger.info(f"Feature matrix built: {weekly_df.shape}")
    return weekly_df, holidays_df


def _get_safe_fill_columns(weekly_df: pd.DataFrame, feat_cfg: dict) -> list:
    """Return list of columns that are safe to bfill/ffill at edges."""
    safe_cols = ["tmean_c"]

    # Temperature lags
    for lag in feat_cfg["temperature"]["lags"]:
        safe_cols.append(f"tmean_lag{lag}")

    # Southern hemisphere columns (anything ending in _CASE, _LAG*, _delta_*)
    sh_cols = [c for c in weekly_df.columns
               if any(c.endswith(suffix) for suffix in ["_ILI_CASE", "_delta_1w"])
               or "_ILI_LAG" in c]
    safe_cols.extend(sh_cols)

    # School features (no NaN expected, but safe)
    safe_cols.extend(["school_in_session", "weeks_since_school"])

    return safe_cols


def add_target(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Add target column (next week's log_ILI)."""
    weekly_df["target"] = weekly_df["log_ILI"].shift(-1)
    weekly_df = weekly_df.dropna(subset=["target"]).reset_index(drop=True)
    return weekly_df


def split_data(weekly_df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    split_ts = pd.Timestamp(split_date)
    train = weekly_df[weekly_df["week_start_date"] <= split_ts].copy()
    test = weekly_df[weekly_df["week_start_date"] > split_ts].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split date {split_date} produced empty train ({len(train)}) or test ({len(test)}) set"
        )

    logger.info(f"Split at {split_date}: train={len(train)} weeks, test={len(test)} weeks")
    return train, test
