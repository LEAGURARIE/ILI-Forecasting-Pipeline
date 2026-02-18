"""
ILI Forecasting Pipeline — Main Runner

Usage:
    python main.py
    python main.py --config config/config.yaml
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.load_data import load_config
from src.features.build_features import build_features, split_data
from src.models.hmm_season import fit_hmm_seasons
from src.models.sarimax_model import SARIMAXModel, auto_tune_sarimax
from src.models.baseline import get_all_baselines
from src.evaluation.metrics import full_evaluation, compare_models, print_evaluation_report
from src.models.next_season_prediction import NextSeasonPredictor
from src.visualization.plots import (
    plot_forecast, plot_actual_vs_predicted, plot_baselines_comparison,
    plot_next_season,
)


def main(config_path: str = "config/config.yaml"):
    """Run the full ILI forecasting pipeline."""

    # ==========================================
    # 1. LOAD CONFIG
    # ==========================================
    print("=" * 60)
    print("ILI FORECASTING PIPELINE")
    print("=" * 60)

    config = load_config(config_path)
    print(f"Config loaded from {config_path}")

    # ==========================================
    # 2. BUILD FEATURES
    # ==========================================
    print("\n[1/6] Building features...")
    weekly_df, holidays_df = build_features(config)
    print(f"  Weekly DF: {weekly_df.shape}")
    print(f"  Holidays:  {len(holidays_df)} rows")

    # ==========================================
    # 3. HMM SEASON DETECTION
    # ==========================================
    print("\n[2/6] Fitting HMM for season detection...")
    weekly_df = fit_hmm_seasons(weekly_df, config)
    n_seasons = weekly_df["season_label"].nunique()
    print(f"  Detected {n_seasons} flu seasons")

    # ==========================================
    # 4. TRAIN/TEST SPLIT
    # ==========================================
    print("\n[3/6] Preparing train/test split...")

    train, test = split_data(weekly_df, config["split"]["split_date"])
    print(f"  Train: {len(train)} weeks")
    print(f"  Test:  {len(test)} weeks")

    # Save processed data
    processed_path = config["paths"]["processed"]
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_parquet(processed_path, index=False)
    print(f"  Saved to {processed_path}")

    # ==========================================
    # 5. BASELINES
    # ==========================================
    print("\n[4/6] Running baseline models...")
    actual_real = np.expm1(test["log_ILI"]).values
    baselines = get_all_baselines(train, test)

    comparison = compare_models(actual_real, baselines, test["is_season"].values)
    print(comparison.to_string())

    # ==========================================
    # 6. SARIMAX
    # ==========================================
    sarimax_cfg = config["sarimax"]

    # Optional: auto-tune
    if sarimax_cfg["auto_arima"]["enabled"]:
        print("\n[5/6] Auto-tuning SARIMAX...")
        best = auto_tune_sarimax(train, config)
        order = best["order"]
        seasonal_order = best["seasonal_order"]
        print(f"  Best order: {order}, seasonal: {seasonal_order}, AIC: {best['aic']:.2f}")
    else:
        order = tuple(sarimax_cfg["order"])
        seasonal_order = tuple(sarimax_cfg["seasonal_order"])

    print(f"\n[5/6] Fitting SARIMAX{order}x{seasonal_order}...")
    model = SARIMAXModel(
        order=order,
        seasonal_order=seasonal_order,
        exog_features=sarimax_cfg["exog_features"],
        maxiter=sarimax_cfg["maxiter"]
    )
    model.fit(train)
    model.summary()

    # Rolling forecast
    print("\n[6/6] Running rolling 1-step-ahead forecast...")
    forecast = model.forecast_rolling(train, test, verbose=True)

    # Add to comparison
    baselines["SARIMAX (rolling)"] = forecast

    # ==========================================
    # 7. EVALUATION
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    results = full_evaluation(
        actual_real, forecast,
        is_season=test["is_season"].values,
        bands=config["evaluation"]["bands"]
    )
    print_evaluation_report(results, "SARIMAX (rolling)")

    # Model comparison table
    print("\n--- Model Comparison ---")
    comparison = compare_models(actual_real, baselines, test["is_season"].values)
    print(comparison.to_string())

    # ==========================================
    # 8. SAVE METRICS, MODEL & PLOT
    # ==========================================
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save model comparison table
    comparison.to_csv(outputs_dir / "model_comparison.csv")
    print(f"  Model comparison saved to {outputs_dir / 'model_comparison.csv'}")

    # Save detailed SARIMAX evaluation
    eval_rows = []
    eval_rows.append({"section": "overall", **results["overall"]})
    eval_rows.append({"section": "overall", "metric": "directional_accuracy",
                      "value": results["directional_accuracy"]})
    for key, val in results.get("band_accuracy", {}).items():
        eval_rows.append({"section": "overall", "metric": key, "value": val})
    for period in ["in_season", "off_season"]:
        if period in results:
            eval_rows.append({"section": period, **results[period]})
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(outputs_dir / "sarimax_evaluation.csv", index=False)
    print(f"  SARIMAX evaluation saved to {outputs_dir / 'sarimax_evaluation.csv'}")

    model_path = Path(config["paths"]["models_dir"]) / "sarimax_best.joblib"
    model.save(str(model_path))

    fig_dir = config["paths"]["figures_dir"]
    plot_forecast(train, test, forecast, "SARIMAX",
                  save_path=f"{fig_dir}/sarimax_forecast.png")
    plot_actual_vs_predicted(actual_real, forecast, "SARIMAX",
                             save_path=f"{fig_dir}/sarimax_scatter.png")
    plot_baselines_comparison(test, baselines,
                              save_path=f"{fig_dir}/model_comparison.png")

    # ==========================================
    # 9. NEXT-SEASON FORECAST
    # ==========================================
    print("\n[7/7] Next-season forecast...")

    # Refit model on FULL data so the internal state is current
    print("  Refitting SARIMAX on full dataset for next-season forecast...")
    full_model = SARIMAXModel(
        order=order,
        seasonal_order=seasonal_order,
        exog_features=sarimax_cfg["exog_features"],
        maxiter=sarimax_cfg["maxiter"]
    )
    full_model.fit(weekly_df)

    predictor = NextSeasonPredictor(
        full_model.results_, sarimax_cfg["exog_features"], weekly_df, config
    )
    future = predictor.forecast(n_weeks=35)
    predictor.summary()

    # Save next-season forecast to CSV (before plots which may block)
    future["future_df"].to_csv(outputs_dir / "next_season_forecast.csv", index=False)
    print(f"  Next-season forecast saved to {outputs_dir / 'next_season_forecast.csv'}")

    plot_next_season(weekly_df, future["future_df"],
                     save_path=f"{fig_dir}/next_season.png")

    print("\nPipeline complete!")
    return weekly_df, model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILI Forecasting Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    args = parser.parse_args()
    main(args.config)
