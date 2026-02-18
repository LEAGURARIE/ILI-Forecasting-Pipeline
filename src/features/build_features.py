"""
Feature engineering pipeline — assembles all features into weekly_df.
"""
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

    # 2. Build continuous weekly grid
    grid = build_continuous_weekly_grid(data_cfg["start_date"], data_cfg["end_date"])

    # 3. Merge and interpolate
    weekly_df = grid.merge(isr_weekly, on="week_start_date", how="left")
    weekly_df["ILI_CASE"] = weekly_df["ILI_CASE"].interpolate(method="linear")

    # 4. Log transform
    weekly_df["log_ILI"] = np.log1p(weekly_df["ILI_CASE"])

    # 5. Delta / momentum features
    weekly_df["ILI_delta_1w"] = weekly_df["ILI_CASE"].diff(1)
    weekly_df["ILI_delta_2w"] = weekly_df["ILI_CASE"].diff(2)
    weekly_df["ILI_pct_1w"] = weekly_df["ILI_CASE"].pct_change(1)
    weekly_df["ILI_rolling_4w"] = weekly_df["ILI_CASE"].rolling(4).mean()

    # 6. Southern Hemisphere regressors
    weekly_df = add_southern_hemisphere_features(
        weekly_df, paths["raw_ili"],
        data_cfg["southern_hemisphere"]["countries"],
        data_cfg["southern_hemisphere"]["lags"]
    )

    # 7. Temperature
    weekly_df = add_temperature_features(
        weekly_df, paths["raw_temperature"],
        feat_cfg["temperature"]["lags"]
    )

    # 8. COVID flags
    covid_start = pd.Timestamp(feat_cfg["covid"]["start"])
    covid_end = pd.Timestamp(feat_cfg["covid"]["end"])
    weekly_df["is_covid"] = (
        (weekly_df["week_start_date"] >= covid_start) &
        (weekly_df["week_start_date"] <= covid_end)
    ).astype(int)
    weekly_df["post_covid"] = (
        weekly_df["week_start_date"] > covid_end
    ).astype(int)

    # 9. School calendar
    weekly_df = add_school_features(weekly_df, feat_cfg["school"])

    # 10. Fill edge NaNs
    weekly_df = weekly_df.bfill().ffill()

    # 11. Holidays (separate DF for Prophet)
    holidays_df = build_holidays_df(
        paths["raw_holidays"],
        feat_cfg["holidays"],
        data_cfg["start_date"],
        data_cfg["end_date"]
    )

    return weekly_df, holidays_df


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
    return train, test
