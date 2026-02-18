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
    mape = np.nanmean(np.abs(
        (actual - predicted) / np.where(actual == 0, np.nan, actual)
    )) * 100
    r2 = r2_score(actual, predicted)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def compute_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of correctly predicted up/down directions."""
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    return np.mean(actual_dir == pred_dir) * 100


def compute_band_accuracy(actual: np.ndarray, predicted: np.ndarray,
                          bands: list = None) -> dict:
    """Percentage of predictions within ±X% of actual."""
    if bands is None:
        bands = [20, 30, 50]

    pct_error = np.abs(
        (actual - predicted) / np.where(actual == 0, np.nan, actual)
    ) * 100

    return {f"within_{b}pct": np.nanmean(pct_error <= b) * 100 for b in bands}


def full_evaluation(actual: np.ndarray, predicted: np.ndarray,
                    is_season: np.ndarray = None, bands: list = None) -> dict:
    """
    Comprehensive evaluation including overall, in-season, off-season.

    Parameters
    ----------
    actual : np.ndarray
    predicted : np.ndarray
    is_season : np.ndarray (optional)
        Binary array: 1 = in-season, 0 = off-season
    bands : list (optional)
        Tolerance bands (default: [20, 30, 50])
    """
    results = {
        "overall": compute_metrics(actual, predicted),
        "directional_accuracy": compute_directional_accuracy(actual, predicted),
        "band_accuracy": compute_band_accuracy(actual, predicted, bands),
    }

    if is_season is not None:
        season_mask = is_season == 1
        offseason_mask = is_season == 0

        if season_mask.sum() > 0:
            results["in_season"] = compute_metrics(
                actual[season_mask], predicted[season_mask]
            )
            results["in_season"]["directional"] = compute_directional_accuracy(
                actual[season_mask], predicted[season_mask]
            )

        if offseason_mask.sum() > 0:
            results["off_season"] = compute_metrics(
                actual[offseason_mask], predicted[offseason_mask]
            )

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


def print_evaluation_report(results: dict, model_name: str = "Model"):
    """Pretty print evaluation results."""
    overall = results["overall"]

    print(f"\n{'=' * 55}")
    print(f"{model_name} — Evaluation Report")
    print(f"{'=' * 55}")
    print(f"  MAPE:     {overall['mape']:.1f}%  ->  Accuracy: {100 - overall['mape']:.1f}%")
    print(f"  MAE:      {overall['mae']:,.0f}")
    print(f"  RMSE:     {overall['rmse']:,.0f}")
    print(f"  R²:       {overall['r2']:.3f}")
    print(f"  Direction: {results['directional_accuracy']:.1f}%")

    bands = results.get("band_accuracy", {})
    for key, val in bands.items():
        print(f"  {key}: {val:.1f}%")

    for period in ["in_season", "off_season"]:
        if period in results:
            m = results[period]
            label = period.replace("_", "-").title()
            print(f"\n  {label}:")
            print(f"    MAE:  {m['mae']:,.0f}")
            print(f"    RMSE: {m['rmse']:,.0f}")
            print(f"    MAPE: {m['mape']:.1f}%")
            if "directional" in m:
                print(f"    Direction: {m['directional']:.1f}%")
