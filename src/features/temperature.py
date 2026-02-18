"""
Temperature feature engineering.
"""
import pandas as pd
from src.data.load_data import load_temperature


def add_temperature_features(
    weekly_df: pd.DataFrame,
    temp_filepath: str,
    lags: list = None
) -> pd.DataFrame:
    """
    Merge weekly temperature and add lags.

    Parameters
    ----------
    weekly_df : pd.DataFrame
    temp_filepath : str
        Path to raw temperature CSV
    lags : list
        Lag weeks to add (e.g., [1, 2])
    """
    if lags is None:
        lags = [1, 2]

    temp = load_temperature(temp_filepath)
    weekly_df = weekly_df.merge(temp, on="week_start_date", how="left")

    # Add lags
    for lag in lags:
        weekly_df[f"tmean_lag{lag}"] = weekly_df["tmean_c"].shift(lag)

    # Fill edge NaNs
    weekly_df["tmean_c"] = weekly_df["tmean_c"].bfill().ffill()
    for lag in lags:
        weekly_df[f"tmean_lag{lag}"] = weekly_df[f"tmean_lag{lag}"].bfill().ffill()

    return weekly_df
