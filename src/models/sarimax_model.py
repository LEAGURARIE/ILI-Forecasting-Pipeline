"""
SARIMAX model — training, hyperparameter tuning, and forecasting.
"""
import logging
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class SARIMAXModel:
    """
    SARIMAX wrapper with rolling one-step-ahead forecast.

    Parameters
    ----------
    order : tuple
        (p, d, q)
    seasonal_order : tuple
        (P, D, Q, m)
    exog_features : list
        Column names of exogenous regressors
    """

    def __init__(self, order: tuple, seasonal_order: tuple, exog_features: list,
                 enforce_stationarity: bool = False, enforce_invertibility: bool = False,
                 maxiter: int = 500):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.exog_features = list(exog_features)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        self.results_ = None

    def fit(self, train: pd.DataFrame, y_col: str = "log_ILI"):
        """Fit SARIMAX on training data."""
        model = SARIMAX(
            train[y_col],
            exog=train[self.exog_features],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = model.fit(disp=False, maxiter=self.maxiter)

        logger.info(
            f"SARIMAX{self.order}x{self.seasonal_order} fitted — "
            f"AIC={self.results_.aic:.1f}"
        )
        return self

    def forecast_multistep(self, test: pd.DataFrame) -> np.ndarray:
        """Multi-step forecast (all test steps at once)."""
        forecast_log = self.results_.forecast(
            steps=len(test),
            exog=test[self.exog_features]
        )
        return np.expm1(forecast_log).clip(min=0).values

    def forecast_rolling(self, train: pd.DataFrame, test: pd.DataFrame,
                         y_col: str = "log_ILI", verbose: bool = True,
                         return_ci: bool = False, alpha: float = 0.05) -> dict | np.ndarray:
        """
        Rolling one-step-ahead forecast using append (no refit).

        At each step, appends the actual observation to the model state
        so the Kalman filter stays current. This is ~100x faster than
        refitting from scratch at each step.

        Parameters
        ----------
        return_ci : bool
            If True, return a dict with 'forecast', 'ci_lower', 'ci_upper'.
            If False, return just the forecast array (backward compatible).
        alpha : float
            Significance level for CI (default 0.05 = 95% CI).
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted before forecasting. Call .fit() first.")

        current_results = self.results_
        predictions = []
        ci_lower_list = []
        ci_upper_list = []
        t0 = time.time()

        for i in range(len(test)):
            next_exog = test[self.exog_features].iloc[i:i + 1].values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = current_results.get_forecast(steps=1, exog=next_exog)

            pred_log = fc.predicted_mean.iloc[0]
            predictions.append(pred_log)

            if return_ci:
                ci_log = fc.conf_int(alpha=alpha).iloc[0]
                ci_lower_list.append(ci_log.iloc[0])
                ci_upper_list.append(ci_log.iloc[1])

            # Append actual observation to advance Kalman filter state (no refit)
            actual_log = test[y_col].iloc[i]
            current_results = current_results.append(
                endog=[actual_log],
                exog=next_exog,
                refit=False,
            )

            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Rolling forecast: {i + 1}/{len(test)} done ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        logger.info(f"  Rolling forecast complete: {len(test)} steps in {elapsed:.1f}s")

        forecast = np.expm1(np.array(predictions)).clip(min=0)

        if return_ci:
            ci_lower = np.expm1(np.array(ci_lower_list)).clip(min=0)
            ci_upper = np.expm1(np.array(ci_upper_list)).clip(min=0)
            return {"forecast": forecast, "ci_lower": ci_lower, "ci_upper": ci_upper}

        return forecast

    def summary(self):
        """Print model summary."""
        if self.results_ is not None:
            print(self.results_.summary())
        else:
            print("Model not fitted yet.")

    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_features": self.exog_features,
            "results": self.results_,
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SARIMAXModel":
        """Load model from disk."""
        data = joblib.load(filepath)
        model = cls(
            order=data["order"],
            seasonal_order=data["seasonal_order"],
            exog_features=data["exog_features"]
        )
        model.results_ = data["results"]
        return model


def auto_tune_sarimax(train: pd.DataFrame, config: dict) -> dict:
    """
    Run auto_arima to find optimal SARIMAX order.

    Returns
    -------
    dict with keys: order, seasonal_order, aic
    """
    import pmdarima as pm

    auto_cfg = config["sarimax"]["auto_arima"]
    exog_features = config["sarimax"]["exog_features"]

    auto_model = pm.auto_arima(
        train["log_ILI"],
        exogenous=train[exog_features],
        seasonal=True,
        m=config["sarimax"]["seasonal_order"][-1],
        max_p=auto_cfg["max_p"],
        max_q=auto_cfg["max_q"],
        max_P=auto_cfg["max_P"],
        max_Q=auto_cfg["max_Q"],
        max_D=auto_cfg["max_D"],
        max_d=1,
        stepwise=auto_cfg["stepwise"],
        suppress_warnings=True,
        error_action="ignore",
        trace=True
    )

    return {
        "order": auto_model.order,
        "seasonal_order": auto_model.seasonal_order,
        "aic": auto_model.aic()
    }


def grid_search_sarimax(train: pd.DataFrame, test: pd.DataFrame,
                         config: dict) -> pd.DataFrame:
    """
    Grid search over ARIMA orders and feature sets.

    Only uses truly exogenous features (temperature-based) — ILI-derived
    features are excluded to prevent target leakage.

    Returns
    -------
    pd.DataFrame ranked by AIC
    """
    exog_features = config["sarimax"]["exog_features"]
    m = config["sarimax"]["seasonal_order"][-1]

    # Only safe (non-leaky) feature sets — no ILI-derived features
    feature_sets = {
        "full": exog_features,
        "temp_only": [f for f in exog_features if "tmean" in f],
        "temp_current": ["tmean_c"],
        "temp_lagged": [f for f in exog_features if "tmean_lag" in f],
    }

    p_range = [2, 3, 4]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]

    results = []
    total = len(p_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range) * len(feature_sets)
    logger.info(f"Grid search: {total} combinations to try")

    for p, q, P, D, Q in product(p_range, q_range, P_range, D_range, Q_range):
        for feat_name, feat_cols in feature_sets.items():
            try:
                t0 = time.time()
                model = SARIMAX(
                    train["log_ILI"],
                    exog=train[feat_cols],
                    order=(p, 1, q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False, maxiter=300)
                fit_time = time.time() - t0

                fcast = np.expm1(
                    res.forecast(steps=len(test), exog=test[feat_cols])
                ).clip(min=0)
                actual = np.expm1(test["log_ILI"])

                results.append({
                    "order": (p, 1, q),
                    "seasonal": (P, D, Q, m),
                    "features": feat_name,
                    "aic": res.aic,
                    "mae": mean_absolute_error(actual, fcast),
                    "rmse": np.sqrt(mean_squared_error(actual, fcast)),
                    "time": fit_time
                })
            except Exception as e:
                logger.debug(f"Grid search failed for ({p},1,{q})x({P},{D},{Q},{m}) "
                             f"features={feat_name}: {e}")
                continue

    result_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    logger.info(f"Grid search complete: {len(result_df)} successful fits out of {total}")
    return result_df
