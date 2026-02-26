"""
Tests for the ILI forecasting pipeline.
"""
import pytest
import pandas as pd
import numpy as np


# ─── School Calendar ──────────────────────────────────────────────────

class TestSchoolCalendar:
    """Tests for school calendar features."""

    @pytest.fixture
    def school_config(self):
        return {
            "summer_start_month": 6, "summer_start_day": 20,
            "summer_end_month": 8, "summer_end_day": 31,
            "school_start_month": 9, "school_start_day": 1
        }

    def test_summer_is_not_in_session(self, school_config):
        from src.features.school_calendar import is_school_in_session
        assert is_school_in_session(pd.Timestamp("2024-07-15"), school_config) == 0
        assert is_school_in_session(pd.Timestamp("2024-08-01"), school_config) == 0
        assert is_school_in_session(pd.Timestamp("2024-06-25"), school_config) == 0

    def test_school_year_is_in_session(self, school_config):
        from src.features.school_calendar import is_school_in_session
        assert is_school_in_session(pd.Timestamp("2024-09-01"), school_config) == 1
        assert is_school_in_session(pd.Timestamp("2024-12-15"), school_config) == 1
        assert is_school_in_session(pd.Timestamp("2024-03-10"), school_config) == 1

    def test_summer_boundary_start(self, school_config):
        from src.features.school_calendar import is_school_in_session
        # June 19 is still in session
        assert is_school_in_session(pd.Timestamp("2024-06-19"), school_config) == 1
        # June 20 is summer
        assert is_school_in_session(pd.Timestamp("2024-06-20"), school_config) == 0

    def test_summer_boundary_end(self, school_config):
        from src.features.school_calendar import is_school_in_session
        # Aug 31 is still summer
        assert is_school_in_session(pd.Timestamp("2024-08-31"), school_config) == 0
        # Sep 1 is school
        assert is_school_in_session(pd.Timestamp("2024-09-01"), school_config) == 1

    def test_weeks_since_school_start_sep1(self, school_config):
        from src.features.school_calendar import weeks_since_school_start
        result = weeks_since_school_start(pd.Timestamp("2024-09-01"), school_config)
        assert result == 0

    def test_weeks_since_school_start_oct1(self, school_config):
        from src.features.school_calendar import weeks_since_school_start
        result = weeks_since_school_start(pd.Timestamp("2024-10-01"), school_config)
        assert 4 < result < 5

    def test_weeks_since_school_start_before_sep(self, school_config):
        from src.features.school_calendar import weeks_since_school_start
        result = weeks_since_school_start(pd.Timestamp("2024-03-01"), school_config)
        assert result > 20

    def test_add_school_features_columns(self, school_config):
        from src.features.school_calendar import add_school_features
        df = pd.DataFrame({
            "week_start_date": pd.date_range("2024-01-07", periods=10, freq="W-SUN")
        })
        result = add_school_features(df, school_config)
        assert "school_in_session" in result.columns
        assert "weeks_since_school" in result.columns
        assert len(result) == 10


# ─── Log Transform ────────────────────────────────────────────────────

class TestTransforms:

    def test_log_transform_roundtrip(self):
        values = np.array([0, 10, 100, 1000, 12000])
        transformed = np.log1p(values)
        recovered = np.expm1(transformed)
        np.testing.assert_allclose(values, recovered, rtol=1e-10)

    def test_log_transform_preserves_order(self):
        values = np.array([10, 100, 1000])
        transformed = np.log1p(values)
        assert np.all(np.diff(transformed) > 0)


# ─── Metrics ──────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.fixture
    def sample_data(self):
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 380, 520])
        return actual, predicted

    def test_compute_metrics_basic(self, sample_data):
        from src.evaluation.metrics import compute_metrics
        actual, predicted = sample_data
        metrics = compute_metrics(actual, predicted)
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]
        assert 0 < metrics["mape"] < 100
        assert 0 < metrics["r2"] <= 1

    def test_compute_metrics_has_smape(self, sample_data):
        from src.evaluation.metrics import compute_metrics
        actual, predicted = sample_data
        metrics = compute_metrics(actual, predicted)
        assert "smape" in metrics
        assert 0 < metrics["smape"] < 200

    def test_compute_metrics_perfect_prediction(self):
        from src.evaluation.metrics import compute_metrics
        actual = np.array([100, 200, 300])
        metrics = compute_metrics(actual, actual)
        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0

    def test_directional_accuracy_perfect(self, sample_data):
        from src.evaluation.metrics import compute_directional_accuracy
        actual, predicted = sample_data
        # Both go up monotonically
        dir_acc = compute_directional_accuracy(actual, predicted)
        assert dir_acc == 100.0

    def test_directional_accuracy_short_array(self):
        from src.evaluation.metrics import compute_directional_accuracy
        result = compute_directional_accuracy(np.array([1]), np.array([2]))
        assert np.isnan(result)

    def test_band_accuracy(self, sample_data):
        from src.evaluation.metrics import compute_band_accuracy
        actual, predicted = sample_data
        bands = compute_band_accuracy(actual, predicted, [20, 50])
        assert "within_20pct" in bands
        assert "within_50pct" in bands
        assert bands["within_50pct"] >= bands["within_20pct"]

    def test_band_accuracy_zero_actuals(self):
        from src.evaluation.metrics import compute_band_accuracy
        actual = np.array([0, 0, 0])
        predicted = np.array([1, 2, 3])
        bands = compute_band_accuracy(actual, predicted, [20])
        assert np.isnan(bands["within_20pct"])

    def test_full_evaluation_structure(self, sample_data):
        from src.evaluation.metrics import full_evaluation
        actual, predicted = sample_data
        is_season = np.array([0, 0, 1, 1, 1])
        results = full_evaluation(actual, predicted, is_season=is_season, bands=[20, 50])

        assert "overall" in results
        assert "in_season" in results
        assert "off_season" in results
        # Normalized: all sections have the same keys
        assert "directional_accuracy" in results["overall"]
        assert "directional_accuracy" in results["in_season"]
        assert "within_20pct" in results["overall"]

    def test_evaluation_to_dataframe(self, sample_data):
        from src.evaluation.metrics import full_evaluation, evaluation_to_dataframe
        actual, predicted = sample_data
        results = full_evaluation(actual, predicted, bands=[20])
        df = evaluation_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert "section" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_compare_models(self, sample_data):
        from src.evaluation.metrics import compare_models
        actual, predicted = sample_data
        models = {
            "model_a": predicted,
            "model_b": actual + 5,  # slightly worse
        }
        comparison = compare_models(actual, models)
        assert "model_a" in comparison.index
        assert "model_b" in comparison.index
        assert "mae" in comparison.columns


# ─── Config Validation ────────────────────────────────────────────────

class TestConfig:

    def test_validate_config_missing_key(self):
        from src.data.load_data import validate_config
        with pytest.raises(KeyError, match="Missing required config key"):
            validate_config({"paths": {}})

    def test_validate_config_bad_date_range(self):
        from src.data.load_data import validate_config
        config = {
            "paths": {
                "raw_ili": "x", "raw_temperature": "x", "raw_holidays": "x",
                "processed": "x", "models_dir": "x", "figures_dir": "x"
            },
            "data": {"country": "ISR", "start_date": "2025-01-01", "end_date": "2020-01-01"},
            "sarimax": {"order": [1, 1, 0], "seasonal_order": [1, 0, 1, 52], "exog_features": []},
            "split": {"split_date": "2023-01-01"},
        }
        with pytest.raises(ValueError, match="start_date.*must be before"):
            validate_config(config)


# ─── Data Split ───────────────────────────────────────────────────────

class TestSplit:

    def test_split_data_basic(self):
        from src.features.build_features import split_data
        df = pd.DataFrame({
            "week_start_date": pd.date_range("2020-01-05", periods=100, freq="W-SUN"),
            "value": range(100)
        })
        train, test = split_data(df, "2021-01-01")
        assert len(train) > 0
        assert len(test) > 0
        assert train["week_start_date"].max() <= pd.Timestamp("2021-01-01")
        assert test["week_start_date"].min() > pd.Timestamp("2021-01-01")

    def test_split_data_empty_raises(self):
        from src.features.build_features import split_data
        df = pd.DataFrame({
            "week_start_date": pd.date_range("2020-01-05", periods=10, freq="W-SUN"),
            "value": range(10)
        })
        with pytest.raises(ValueError, match="empty train"):
            split_data(df, "2019-01-01")


# ─── HMM Season Detection ────────────────────────────────────────────

class TestHMM:

    def test_find_blocks(self):
        from src.models.hmm_season import _find_blocks
        arr = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
        blocks = _find_blocks(arr, target=1)
        assert blocks == [(1, 3), (6, 7)]

    def test_find_blocks_empty(self):
        from src.models.hmm_season import _find_blocks
        arr = np.array([0, 0, 0])
        assert _find_blocks(arr, target=1) == []

    def test_find_blocks_all_ones(self):
        from src.models.hmm_season import _find_blocks
        arr = np.array([1, 1, 1])
        assert _find_blocks(arr, target=1) == [(0, 2)]

    def test_filter_short_seasons(self):
        from src.models.hmm_season import _filter_short_seasons
        df = pd.DataFrame({
            "is_season": [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],  # block of 2, block of 5
        })
        result = _filter_short_seasons(df, min_weeks=4)
        # First block (2 weeks) should be removed
        assert result["is_season"].iloc[1] == 0
        assert result["is_season"].iloc[2] == 0
        # Second block (5 weeks) should remain
        assert result["is_season"].iloc[4:9].sum() == 5


# ─── Model Save/Load ─────────────────────────────────────────────────

class TestModelIO:

    def test_sarimax_model_init(self):
        from src.models.sarimax_model import SARIMAXModel
        model = SARIMAXModel(
            order=(3, 1, 0),
            seasonal_order=(1, 0, 1, 52),
            exog_features=["tmean_c"]
        )
        assert model.order == (3, 1, 0)
        assert model.exog_features == ["tmean_c"]
        assert model.results_ is None

    def test_forecast_before_fit_raises(self):
        from src.models.sarimax_model import SARIMAXModel
        model = SARIMAXModel(
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 1),
            exog_features=[]
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.forecast_rolling(
                pd.DataFrame({"log_ILI": [1, 2]}),
                pd.DataFrame({"log_ILI": [3]})
            )
