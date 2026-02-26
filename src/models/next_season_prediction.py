"""
Next-season prediction using a fitted SARIMAX model.

Uses rolling one-step-ahead forecasting: at each step the prediction is
appended back to the model state so the AR and seasonal components stay
active across the full horizon.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from epiweeks import Week

from src.features.school_calendar import is_school_in_session, weeks_since_school_start

logger = logging.getLogger(__name__)

# Features that can be computed exactly for future dates
# (calendar-based or fixed historical flags)
EXACT_FEATURES = {"weeks_since_school", "school_in_session", "is_covid", "post_covid"}


class NextSeasonPredictor:

    def __init__(self, model_results, exog_features: list,
                 weekly_df: pd.DataFrame, config: dict):
        self.results_ = model_results
        self.exog_features = list(exog_features)
        self.weekly_df = weekly_df
        self.config = config

        # Historical medians by epi-week — used only for features
        # that can't be computed exactly (temperature, Australia ILI)
        median_features = [f for f in self.exog_features if f not in EXACT_FEATURES]
        self._hist_by_week = (
            weekly_df.groupby("week")[median_features].median()
        )
        self._global_median = weekly_df[median_features].median()

        # School config for exact computation
        self._school_config = config.get("features", {}).get("school", {})
        # COVID config for exact computation
        self._covid_end = pd.Timestamp(
            config.get("features", {}).get("covid", {}).get("end", "2021-06-30")
        )

    def _get_exog_for_date(self, future_date: pd.Timestamp, epi_week: int) -> pd.DataFrame:
        """
        Build a 1-row exogenous DataFrame for a future date.

        - Calendar/COVID features: computed exactly from the date
        - Temperature/Australia features: historical median by epi-week
        """
        row = {}

        for feat in self.exog_features:
            if feat == "weeks_since_school":
                row[feat] = weeks_since_school_start(future_date, self._school_config)

            elif feat == "school_in_session":
                row[feat] = is_school_in_session(future_date, self._school_config)

            elif feat == "is_covid":
                # Future dates are never in the COVID period
                row[feat] = 0

            elif feat == "post_covid":
                # Future dates are always post-COVID
                row[feat] = 1 if future_date > self._covid_end else 0

            else:
                # Temperature, Australia ILI, etc. — use historical median
                if epi_week in self._hist_by_week.index and feat in self._hist_by_week.columns:
                    row[feat] = self._hist_by_week.loc[epi_week, feat]
                elif feat in self._global_median.index:
                    row[feat] = self._global_median[feat]
                else:
                    row[feat] = 0.0
                    logger.warning(f"No historical data for feature '{feat}', using 0.0")

        return pd.DataFrame([row], columns=self.exog_features)

    def forecast(self, n_weeks: int = 35) -> dict:
        last_date = self.weekly_df["week_start_date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=n_weeks,
            freq="W-SUN",
        )

        current_results = self.results_
        predicted_list = []
        ci_lower_list = []
        ci_upper_list = []

        logger.info(f"Rolling 1-step-ahead forecast ({n_weeks} weeks) ...")

        for i, fdate in enumerate(future_dates):
            epi_week = Week.fromdate(fdate).week
            next_exog = self._get_exog_for_date(fdate, epi_week)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = current_results.get_forecast(steps=1, exog=next_exog)

            pred_log = fc.predicted_mean.iloc[0]
            ci_log = fc.conf_int(alpha=0.05).iloc[0]

            pred_ili = max(0.0, np.expm1(pred_log))
            ci_lo = max(0.0, np.expm1(ci_log.iloc[0]))
            ci_hi = max(0.0, np.expm1(ci_log.iloc[1]))

            predicted_list.append(pred_ili)
            ci_lower_list.append(ci_lo)
            ci_upper_list.append(ci_hi)

            # Append prediction as "observation" to advance the
            # Kalman filter state. Keeps AR(3) and SAR(1)x52
            # fed with recent values across the full horizon.
            current_results = current_results.append(
                endog=[pred_log],
                exog=next_exog.values,
                refit=False,
            )

            print(
                f"  Week {i + 1:2d}/{n_weeks}  {fdate.strftime('%Y-%m-%d')}  "
                f"epi={epi_week:2d}  ->  ILI = {pred_ili:>8,.0f}  "
                f"({ci_lo:>7,.0f} - {ci_hi:>7,.0f})"
            )

        predicted = np.array(predicted_list)
        ci_lower = np.array(ci_lower_list)
        ci_upper = np.array(ci_upper_list)
        peak_idx = int(np.argmax(predicted))

        future_df = pd.DataFrame({
            "week_start_date": future_dates,
            "predicted_ILI": predicted,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "week": [Week.fromdate(d).week for d in future_dates],
        })

        self._last_forecast = {
            "dates": future_dates,
            "predicted": predicted,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "peak_idx": peak_idx,
            "future_df": future_df,
        }
        return self._last_forecast

    def summary(self) -> None:
        if not hasattr(self, "_last_forecast"):
            print("No forecast available. Call .forecast() first.")
            return

        f = self._last_forecast
        dates, predicted = f["dates"], f["predicted"]
        ci_lower, ci_upper = f["ci_lower"], f["ci_upper"]
        peak_idx = f["peak_idx"]

        print(f"\n{'=' * 60}")
        print("NEXT SEASON PREDICTION SUMMARY (Rolling 1-Step)")
        print(f"{'=' * 60}")
        print(f"Forecast period:  {dates[0].strftime('%d %b %Y')} -> "
              f"{dates[-1].strftime('%d %b %Y')}")
        print(f"Predicted peak:   {predicted[peak_idx]:,.0f} cases")
        print(f"Peak 95% CI:      {ci_lower[peak_idx]:,.0f} - "
              f"{ci_upper[peak_idx]:,.0f}")
        print(f"Peak week:        {dates[peak_idx].strftime('%d %b %Y')}")
        print(f"Total predicted:  {predicted.sum():,.0f} cases")

        season_data = self.weekly_df[self.weekly_df["season_label"].notna()].copy()
        seasons = sorted(season_data["season_label"].dropna().unique())

        # Exclude incomplete trailing season
        last_data_date = self.weekly_df["week_start_date"].max()
        if len(seasons) > 0:
            last_season = season_data[season_data["season_label"] == seasons[-1]]
            last_season_end = last_season["week_start_date"].max()
            if last_season_end >= last_data_date and len(last_season) < 10:
                seasons = seasons[:-1]

        recent_seasons = seasons[-3:] if len(seasons) >= 3 else seasons

        print("\nHistorical comparison:")
        for season in recent_seasons:
            s = season_data[season_data["season_label"] == season]
            if len(s) > 0:
                print(f"  Season {int(season)}: peak={s['ILI_CASE'].max():,.0f}, "
                      f"total={s['ILI_CASE'].sum():,.0f}, weeks={len(s)}")


def run_next_season(model, weekly_df: pd.DataFrame, config: dict,
                    n_weeks: int = 35) -> dict:
    predictor = NextSeasonPredictor(
        model_results=model.results_,
        exog_features=model.exog_features,
        weekly_df=weekly_df,
        config=config,
    )
    result = predictor.forecast(n_weeks=n_weeks)
    predictor.summary()
    return result
