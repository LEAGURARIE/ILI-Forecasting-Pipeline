"""
Southern Hemisphere ILI regressors (leading indicators).
"""
import pandas as pd
from src.data.load_data import load_southern_hemisphere


def add_southern_hemisphere_features(
    weekly_df: pd.DataFrame,
    ili_filepath: str,
    countries: list = None,
    lags: list = None
) -> pd.DataFrame:
    """
    Add Southern Hemisphere ILI and delta features.

    Parameters
    ----------
    weekly_df : pd.DataFrame
    ili_filepath : str
        Path to WHO FluID CSV
    countries : list
        ISO3 codes (e.g., ["AUS"])
    lags : list
        Lag weeks (e.g., [1])
    """
    if countries is None:
        countries = ["AUS"]
    if lags is None:
        lags = [1]

    sh_data = load_southern_hemisphere(ili_filepath, countries)
    weekly_df = weekly_df.merge(sh_data, on="week_start_date", how="left")

    for code in countries:
        col = f"{code}_ILI_CASE"

        # Interpolate missing
        weekly_df[col] = weekly_df[col].interpolate(method="linear").fillna(0)

        # Add lags
        for lag in lags:
            weekly_df[f"{code}_ILI_LAG{lag}"] = weekly_df[col].shift(lag)

        # Delta
        weekly_df[f"{code}_delta_1w"] = weekly_df[col].diff(1)

    # Fill edge NaNs
    weekly_df = weekly_df.bfill().ffill()

    return weekly_df
