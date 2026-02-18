"""
SARIMAX model — training, hyperparameter tuning, and forecasting.
"""
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
        self.exog_features = exog_features
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
        return self

    def forecast_multistep(self, test: pd.DataFrame) -> np.ndarray:
        """Multi-step forecast (all test steps at once)."""
        forecast_log = self.results_.forecast(
            steps=len(test),
            exog=test[self.exog_features]
        )
        return np.expm1(forecast_log).clip(min=0).values

    def forecast_rolling(self, train: pd.DataFrame, test: pd.DataFrame,
                         y_col: str = "log_ILI", verbose: bool = True) -> np.ndarray:
        """
        Rolling one-step-ahead forecast.
        Refits model at each step with actual values added to history.
        """
        history_y = train[y_col].tolist()
        history_exog = train[self.exog_features].values.tolist()
        predictions = []

        for i in range(len(test)):
            model = SARIMAX(
                history_y,
                exog=history_exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(disp=False, maxiter=200)

            next_exog = test[self.exog_features].iloc[i:i + 1].values
            pred = res.forecast(steps=1, exog=next_exog)
            predictions.append(pred[0])

            # Add actual value to history
            history_y.append(test[y_col].iloc[i])
            history_exog.append(test[self.exog_features].iloc[i].tolist())

            if verbose and (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(test)} done")

        return np.expm1(np.array(predictions)).clip(min=0)

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
        print(f"Model saved to {filepath}")

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

    Returns
    -------
    pd.DataFrame ranked by AIC
    """
    from itertools import product
    import time

    exog_features = config["sarimax"]["exog_features"]
    m = config["sarimax"]["seasonal_order"][-1]

    feature_sets = {
        "full": exog_features,
        "temp_only": [f for f in exog_features if "tmean" in f],
        "momentum_only": [f for f in exog_features if "ILI" in f],
        "minimal": ["tmean_lag1", "ILI_pct_1w", "ILI_rolling_4w"],
    }

    p_range = [2, 3, 4]
    q_range = [0, 1, 2]
    P_range = [0, 1]
    D_range = [0, 1]
    Q_range = [0, 1]

    results = []

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

                from sklearn.metrics import mean_absolute_error, mean_squared_error

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
            except Exception:
                continue

    return pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
