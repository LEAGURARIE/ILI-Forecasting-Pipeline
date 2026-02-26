"""
Evaluation metrics and model comparison.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE — guard against zero actuals
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(
            (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
        )) * 100
    else:
        mape = np.nan

    # SMAPE — symmetric, more robust when actuals are near zero
    denom = np.abs(actual) + np.abs(predicted)
    smape_values = np.where(
        denom == 0, 0.0,
        2 * np.abs(actual - predicted) / denom
    )
    smape = np.mean(smape_values) * 100

    r2 = r2_score(actual, predicted)

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2}


def compute_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of correctly predicted up/down directions."""
    if len(actual) < 2:
        return np.nan
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    return np.mean(actual_dir == pred_dir) * 100


def compute_band_accuracy(actual: np.ndarray, predicted: np.ndarray,
                          bands: list = None) -> dict:
    """Percentage of predictions within ±X% of actual."""
    if bands is None:
        bands = [20, 30, 50]

    nonzero_mask = actual != 0
    if nonzero_mask.sum() == 0:
        return {f"within_{b}pct": np.nan for b in bands}

    pct_error = np.abs(
        (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
    ) * 100

    return {f"within_{b}pct": np.mean(pct_error <= b) * 100 for b in bands}


def full_evaluation(actual: np.ndarray, predicted: np.ndarray,
                    is_season: np.ndarray = None, bands: list = None) -> dict:
    """
    Comprehensive evaluation including overall, in-season, off-season.

    Returns a normalized structure where each section has the same metric keys.

    Parameters
    ----------
    actual : np.ndarray
    predicted : np.ndarray
    is_season : np.ndarray (optional)
        Binary array: 1 = in-season, 0 = off-season
    bands : list (optional)
        Tolerance bands (default: [20, 30, 50])
    """
    overall_metrics = compute_metrics(actual, predicted)
    overall_metrics["directional_accuracy"] = compute_directional_accuracy(actual, predicted)
    overall_metrics.update(compute_band_accuracy(actual, predicted, bands))

    results = {"overall": overall_metrics}

    if is_season is not None:
        season_mask = is_season == 1
        offseason_mask = is_season == 0

        if season_mask.sum() > 0:
            in_season = compute_metrics(actual[season_mask], predicted[season_mask])
            in_season["directional_accuracy"] = compute_directional_accuracy(
                actual[season_mask], predicted[season_mask]
            )
            in_season.update(compute_band_accuracy(actual[season_mask], predicted[season_mask], bands))
            results["in_season"] = in_season

        if offseason_mask.sum() > 0:
            off_season = compute_metrics(actual[offseason_mask], predicted[offseason_mask])
            off_season["directional_accuracy"] = compute_directional_accuracy(
                actual[offseason_mask], predicted[offseason_mask]
            )
            off_season.update(compute_band_accuracy(actual[offseason_mask], predicted[offseason_mask], bands))
            results["off_season"] = off_season

    return results


def compare_models(actual: np.ndarray, models: dict,
                   is_season: np.ndarray = None) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Parameters
    ----------
    actual : np.ndarray
    models : dict
        {model_name: predictions_array}
    is_season : np.ndarray (optional)

    Returns
    -------
    pd.DataFrame with metrics for each model
    """
    rows = []
    for name, preds in models.items():
        metrics = compute_metrics(actual, preds)
        metrics["model"] = name
        metrics["dir_accuracy"] = compute_directional_accuracy(actual, preds)

        if is_season is not None:
            mask = is_season == 1
            if mask.sum() > 0:
                season_m = compute_metrics(actual[mask], preds[mask])
                metrics["mae_season"] = season_m["mae"]
                metrics["rmse_season"] = season_m["rmse"]

        rows.append(metrics)

    return pd.DataFrame(rows).set_index("model")


def evaluation_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert the nested evaluation results dict to a flat DataFrame.

    Useful for saving to CSV.
    """
    rows = []
    for section, metrics in results.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                rows.append({"section": section, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def print_evaluation_report(results: dict, model_name: str = "Model"):
    """Pretty print evaluation results."""
    overall = results["overall"]

    print(f"\n{'=' * 55}")
    print(f"{model_name} — Evaluation Report")
    print(f"{'=' * 55}")
    print(f"  MAPE:     {overall['mape']:.1f}%  ->  Accuracy: {100 - overall['mape']:.1f}%")
    print(f"  SMAPE:    {overall['smape']:.1f}%")
    print(f"  MAE:      {overall['mae']:,.0f}")
    print(f"  RMSE:     {overall['rmse']:,.0f}")
    print(f"  R²:       {overall['r2']:.3f}")
    print(f"  Direction: {overall['directional_accuracy']:.1f}%")

    for key, val in overall.items():
        if key.startswith("within_"):
            print(f"  {key}: {val:.1f}%")

    for period in ["in_season", "off_season"]:
        if period in results:
            m = results[period]
            label = period.replace("_", "-").title()
            print(f"\n  {label}:")
            print(f"    MAE:  {m['mae']:,.0f}")
            print(f"    RMSE: {m['rmse']:,.0f}")
            print(f"    MAPE: {m['mape']:.1f}%")
            print(f"    SMAPE: {m['smape']:.1f}%")
            if "directional_accuracy" in m:
                print(f"    Direction: {m['directional_accuracy']:.1f}%")


# ---------------------------------------------------------------------------
# HMM Season Evaluation
# ---------------------------------------------------------------------------

def compute_hmm_season_stats(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-season statistics from HMM-labelled weekly data.

    Returns a DataFrame with one row per season containing:
    season_label, onset_date, onset_week, peak_date, peak_week,
    end_date, end_week, duration_weeks, peak_ili, total_ili_burden
    """
    season_data = weekly_df[weekly_df["season_label"].notna()].copy()
    seasons = sorted(season_data["season_label"].unique())

    rows = []
    for season in seasons:
        s = season_data[season_data["season_label"] == season].sort_values("week_start_date")
        onset_date = s["week_start_date"].iloc[0]
        end_date = s["week_start_date"].iloc[-1]
        peak_idx = s["ILI_CASE"].idxmax()
        peak_date = s.loc[peak_idx, "week_start_date"]

        rows.append({
            "season_label": int(season),
            "onset_date": onset_date,
            "onset_week": onset_date.isocalendar()[1],
            "peak_date": peak_date,
            "peak_week": peak_date.isocalendar()[1],
            "end_date": end_date,
            "end_week": end_date.isocalendar()[1],
            "duration_weeks": len(s),
            "peak_ili": s["ILI_CASE"].max(),
            "total_ili_burden": s["ILI_CASE"].sum(),
        })

    return pd.DataFrame(rows)


def compute_hmm_cross_season_summary(season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics (mean/std/min/max) across seasons.

    Parameters
    ----------
    season_stats : pd.DataFrame
        Output of compute_hmm_season_stats()

    Returns
    -------
    pd.DataFrame with rows for each aggregate metric
    """
    numeric_cols = ["duration_weeks", "onset_week", "peak_week", "peak_ili", "total_ili_burden"]
    agg_funcs = ["mean", "std", "min", "max"]

    rows = []
    for col in numeric_cols:
        for func in agg_funcs:
            rows.append({
                "metric": col,
                "statistic": func,
                "value": season_stats[col].agg(func),
            })

    return pd.DataFrame(rows)


def compute_hmm_model_quality(hmm_model, X: np.ndarray) -> dict:
    """
    Compute HMM model quality metrics.

    Parameters
    ----------
    hmm_model : GaussianHMM
        Fitted HMM model
    X : np.ndarray
        Observation data (n_samples, 1)

    Returns
    -------
    dict with log_likelihood, AIC, BIC, state_means, state_covariances,
    transition_matrix
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]
    log_likelihood = hmm_model.score(X)

    # Number of free parameters for a 2-state Gaussian HMM:
    # transition matrix: n*(n-1), means: n*d, covariances: n*d (diag) or n*d*d (full)
    n_components = hmm_model.n_components
    n_features = X.shape[1]
    n_params = (n_components * (n_components - 1)  # transition
                + n_components * n_features          # means
                + n_components * n_features)          # covariances (diag)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return {
        "log_likelihood": log_likelihood,
        "AIC": aic,
        "BIC": bic,
        "state_means": hmm_model.means_.flatten().tolist(),
        "state_covariances": hmm_model.covars_.flatten().tolist(),
        "transition_matrix": hmm_model.transmat_.tolist(),
    }


def print_hmm_season_report(season_stats: pd.DataFrame,
                             cross_season: pd.DataFrame,
                             model_quality: dict):
    """Pretty print HMM season detection report."""
    print(f"\n{'=' * 55}")
    print("HMM Season Detection Report")
    print(f"{'=' * 55}")

    # Per-season summary
    print(f"\n  Detected {len(season_stats)} seasons:")
    for _, row in season_stats.iterrows():
        print(f"    Season {int(row['season_label'])}: "
              f"{row['onset_date'].strftime('%Y-%m-%d')} -> "
              f"{row['end_date'].strftime('%Y-%m-%d')}  "
              f"({int(row['duration_weeks'])} wks, "
              f"peak={row['peak_ili']:,.0f})")

    # Cross-season summary
    print(f"\n  Cross-Season Summary:")
    for _, row in cross_season.iterrows():
        print(f"    {row['metric']:20s} {row['statistic']:5s}: {row['value']:>10.1f}")

    # Model quality
    print(f"\n  Model Quality:")
    print(f"    Log-likelihood: {model_quality['log_likelihood']:,.2f}")
    print(f"    AIC:            {model_quality['AIC']:,.2f}")
    print(f"    BIC:            {model_quality['BIC']:,.2f}")
    print(f"    State means:    {model_quality['state_means']}")
    print(f"    Transition matrix:")
    for i, row in enumerate(model_quality['transition_matrix']):
        print(f"      State {i}: [{', '.join(f'{v:.3f}' for v in row)}]")
