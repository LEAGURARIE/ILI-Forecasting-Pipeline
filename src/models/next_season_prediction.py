"""
Next-season prediction using a fitted SARIMAX model.

Uses rolling one-step-ahead forecasting: at each step the prediction is
appended back to the model state so the AR and seasonal components stay
active across the full horizon.
"""
import warnings
import numpy as np
import pandas as pd


class NextSeasonPredictor:

    def __init__(self, model_results, exog_features: list,
                 weekly_df: pd.DataFrame, config: dict):
        self.results_ = model_results
        self.exog_features = list(exog_features)
        self.weekly_df = weekly_df
        self.config = config

        self._hist_by_week = (
            weekly_df.groupby("week")[self.exog_features].median()
        )
        self._global_median = weekly_df[self.exog_features].median()

    def _get_exog_for_week(self, epi_week: int) -> pd.DataFrame:
        if epi_week in self._hist_by_week.index:
            return self._hist_by_week.loc[[epi_week]]
        return pd.DataFrame([self._global_median], columns=self.exog_features)

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

        print(f"Rolling 1-step-ahead forecast ({n_weeks} weeks) ...")

        for i, fdate in enumerate(future_dates):
            epi_week = fdate.isocalendar().week
            next_exog = self._get_exog_for_week(epi_week)

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

            # ── KEY FIX ──────────────────────────────────────
            # Append prediction as "observation" to advance the
            # Kalman filter state. Keeps AR(3) and SAR(1)x52
            # fed with recent values across the full horizon.
            # ─────────────────────────────────────────────────
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
            "week": [d.isocalendar().week for d in future_dates],
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