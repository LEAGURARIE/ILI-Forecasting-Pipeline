"""
Baseline models for comparison.
"""
import numpy as np
import pandas as pd


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Next week = this week (last known value)."""
    shifted = test["log_ILI"].shift(1).copy()
    shifted.iloc[0] = train["log_ILI"].iloc[-1]
    return np.expm1(shifted).values


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Next week = same week last year."""
    predictions = []
    for _, row in test.iterrows():
        same_week = train[
            (train["week"] == row["week"]) &
            (train["year"] == row["year"] - 1)
        ]["ILI_CASE"]

        if len(same_week) > 0:
            predictions.append(same_week.values[0])
        else:
            fallback = train[train["week"] == row["week"]]["ILI_CASE"].mean()
            predictions.append(fallback)

    return np.array(predictions)


def weekly_mean_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical mean of same epidemiological week."""
    week_means = train.groupby("week")["ILI_CASE"].mean()
    return test["week"].map(week_means).values


def weekly_median_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical median of same epidemiological week."""
    week_medians = train.groupby("week")["ILI_CASE"].median()
    return test["week"].map(week_medians).values


def get_all_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Run all baseline models and return predictions."""
    return {
        "Naive (last week)": naive_forecast(train, test),
        "Seasonal Naive (ly)": seasonal_naive_forecast(train, test),
        "Weekly Mean": weekly_mean_forecast(train, test),
        "Weekly Median": weekly_median_forecast(train, test),
    }
