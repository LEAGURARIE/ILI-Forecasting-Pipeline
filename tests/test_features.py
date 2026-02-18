"""
Basic tests for feature engineering pipeline.
"""
import pytest
import pandas as pd
import numpy as np


def test_school_in_session():
    """Test school calendar logic."""
    from src.features.school_calendar import is_school_in_session

    config = {
        "summer_start_month": 6, "summer_start_day": 20,
        "summer_end_month": 8, "summer_end_day": 31,
        "school_start_month": 9, "school_start_day": 1
    }

    # Summer — should be 0
    assert is_school_in_session(pd.Timestamp("2024-07-15"), config) == 0
    assert is_school_in_session(pd.Timestamp("2024-08-01"), config) == 0
    assert is_school_in_session(pd.Timestamp("2024-06-25"), config) == 0

    # School year — should be 1
    assert is_school_in_session(pd.Timestamp("2024-09-01"), config) == 1
    assert is_school_in_session(pd.Timestamp("2024-12-15"), config) == 1
    assert is_school_in_session(pd.Timestamp("2024-03-10"), config) == 1


def test_weeks_since_school():
    """Test weeks_since_school_start calculation."""
    from src.features.school_calendar import weeks_since_school_start

    config = {"school_start_month": 9, "school_start_day": 1}

    # Exactly Sep 1 → 0 weeks
    result = weeks_since_school_start(pd.Timestamp("2024-09-01"), config)
    assert result == 0

    # Oct 1 → ~4.3 weeks
    result = weeks_since_school_start(pd.Timestamp("2024-10-01"), config)
    assert 4 < result < 5

    # Before Sep 1 → uses previous year
    result = weeks_since_school_start(pd.Timestamp("2024-03-01"), config)
    assert result > 20


def test_log_transform_roundtrip():
    """Test that log1p/expm1 roundtrip preserves values."""
    values = np.array([0, 10, 100, 1000, 12000])
    transformed = np.log1p(values)
    recovered = np.expm1(transformed)
    np.testing.assert_allclose(values, recovered, rtol=1e-10)


def test_metrics():
    """Test evaluation metrics."""
    from src.evaluation.metrics import compute_metrics, compute_directional_accuracy

    actual = np.array([100, 200, 300, 400, 500])
    predicted = np.array([110, 190, 310, 380, 520])

    metrics = compute_metrics(actual, predicted)
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert 0 < metrics["mape"] < 100
    assert 0 < metrics["r2"] <= 1

    # Perfect directional accuracy
    dir_acc = compute_directional_accuracy(actual, predicted)
    assert dir_acc == 100.0
