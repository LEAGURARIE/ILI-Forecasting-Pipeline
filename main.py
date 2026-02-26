"""
ILI Forecasting Pipeline — Main Runner

Usage:
    python main.py
    python main.py --config config/config.yaml
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.load_data import load_config
from src.features.build_features import build_features, split_data
from src.models.hmm_season import fit_hmm_seasons
from src.models.sarimax_model import SARIMAXModel, auto_tune_sarimax
from src.models.baseline import get_all_baselines
from src.evaluation.metrics import (
    full_evaluation, compare_models, print_evaluation_report, evaluation_to_dataframe,
    compute_hmm_season_stats, compute_hmm_cross_season_summary,
    compute_hmm_model_quality, print_hmm_season_report,
)
from src.models.next_season_prediction import NextSeasonPredictor
from src.visualization.plots import (
    plot_forecast, plot_actual_vs_predicted, plot_baselines_comparison,
    plot_next_season, plot_residual_diagnostics, plot_hmm_seasonality,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def main(config_path: str = "config/config.yaml"):
    """Run the full ILI forecasting pipeline."""

    setup_logging()

    # ==========================================
    # 1. LOAD CONFIG
    # ==========================================
    logger.info("=" * 60)
    logger.info("ILI FORECASTING PIPELINE")
    logger.info("=" * 60)

    config = load_config(config_path)
    logger.info(f"Config loaded from {config_path}")

    # ==========================================
    # 2. BUILD FEATURES
    # ==========================================
    logger.info("[1/7] Building features...")
    weekly_df, holidays_df = build_features(config)
    logger.info(f"  Weekly DF: {weekly_df.shape}")
    logger.info(f"  Holidays:  {len(holidays_df)} rows")

    # ==========================================
    # 3. HMM SEASON DETECTION
    # ==========================================
    logger.info("[2/7] Fitting HMM for season detection...")
    weekly_df, hmm_model = fit_hmm_seasons(weekly_df, config)
    n_seasons = weekly_df["season_label"].nunique()
    logger.info(f"  Detected {n_seasons} flu seasons")

    # HMM season metrics
    hmm_season_stats = compute_hmm_season_stats(weekly_df)
    hmm_cross_season = compute_hmm_cross_season_summary(hmm_season_stats)
    hmm_quality = compute_hmm_model_quality(
        hmm_model, weekly_df["ILI_CASE"].values.reshape(-1, 1)
    )
    print_hmm_season_report(hmm_season_stats, hmm_cross_season, hmm_quality)

    # ==========================================
    # 4. TRAIN/TEST SPLIT
    # ==========================================
    logger.info("[3/7] Preparing train/test split...")

    train, test = split_data(weekly_df, config["split"]["split_date"])

    # Save processed data
    processed_path = config["paths"]["processed"]
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_parquet(processed_path, index=False)
    logger.info(f"  Saved to {processed_path}")

    # ==========================================
    # 5. BASELINES
    # ==========================================
    logger.info("[4/7] Running baseline models...")
    actual_real = np.expm1(test["log_ILI"]).values
    baselines = get_all_baselines(train, test)

    # ==========================================
    # 6. SARIMAX
    # ==========================================
    sarimax_cfg = config["sarimax"]

    # Optional: auto-tune
    if sarimax_cfg["auto_arima"]["enabled"]:
        logger.info("[5/7] Auto-tuning SARIMAX...")
        best = auto_tune_sarimax(train, config)
        order = best["order"]
        seasonal_order = best["seasonal_order"]
        logger.info(f"  Best order: {order}, seasonal: {seasonal_order}, AIC: {best['aic']:.2f}")
    else:
        order = tuple(sarimax_cfg["order"])
        seasonal_order = tuple(sarimax_cfg["seasonal_order"])

    logger.info(f"[5/7] Fitting SARIMAX{order}x{seasonal_order}...")
    model = SARIMAXModel(
        order=order,
        seasonal_order=seasonal_order,
        exog_features=sarimax_cfg["exog_features"],
        maxiter=sarimax_cfg["maxiter"]
    )
    model.fit(train)
    model.summary()

    # Rolling forecast
    logger.info("[6/7] Running rolling 1-step-ahead forecast...")
    forecast_result = model.forecast_rolling(train, test, verbose=True, return_ci=True)
    forecast = forecast_result["forecast"]
    ci_lower = forecast_result["ci_lower"]
    ci_upper = forecast_result["ci_upper"]

    # Add to comparison
    baselines["SARIMAX (rolling)"] = forecast

    # ==========================================
    # 7. EVALUATION
    # ==========================================
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

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
    # 8. SAVE METRICS, MODEL & PLOTS
    # ==========================================
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config["paths"]["figures_dir"]

    # Save model comparison table
    comparison.to_csv(outputs_dir / "model_comparison.csv")
    logger.info(f"  Model comparison saved to {outputs_dir / 'model_comparison.csv'}")

    # Save HMM season summary
    hmm_season_stats.to_csv(outputs_dir / "hmm_season_summary.csv", index=False)
    logger.info(f"  HMM season summary saved to {outputs_dir / 'hmm_season_summary.csv'}")

    # Save detailed SARIMAX evaluation (normalized structure)
    eval_df = evaluation_to_dataframe(results)
    eval_df.to_csv(outputs_dir / "sarimax_evaluation.csv", index=False)
    logger.info(f"  SARIMAX evaluation saved to {outputs_dir / 'sarimax_evaluation.csv'}")

    # Save model
    model_path = Path(config["paths"]["models_dir"]) / "sarimax_best.joblib"
    model.save(str(model_path))

    # Plots — wrapped in try/except so a plotting error doesn't crash the pipeline
    try:
        plot_forecast(train, test, forecast, "SARIMAX",
                      ci_lower=ci_lower, ci_upper=ci_upper,
                      save_path=f"{fig_dir}/sarimax_forecast.png")
        plot_actual_vs_predicted(actual_real, forecast, "SARIMAX",
                                 save_path=f"{fig_dir}/sarimax_scatter.png")
        plot_baselines_comparison(test, baselines,
                                  save_path=f"{fig_dir}/model_comparison.png")
        plot_residual_diagnostics(actual_real, forecast,
                                  dates=test["week_start_date"],
                                  model_name="SARIMAX",
                                  save_path=f"{fig_dir}/sarimax_residuals.png")
        plot_hmm_seasonality(weekly_df,
                             save_path=f"{fig_dir}/hmm_seasonality.png")
        logger.info("  All plots saved")
    except Exception as e:
        logger.warning(f"  Plot generation failed (non-fatal): {e}")

    # ==========================================
    # 9. NEXT-SEASON FORECAST
    # ==========================================
    logger.info("[7/7] Next-season forecast...")

    # Refit model on FULL data so the internal state is current
    logger.info("  Refitting SARIMAX on full dataset for next-season forecast...")
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

    # Save next-season forecast to CSV
    future["future_df"].to_csv(outputs_dir / "next_season_forecast.csv", index=False)
    logger.info(f"  Next-season forecast saved to {outputs_dir / 'next_season_forecast.csv'}")

    try:
        plot_next_season(weekly_df, future["future_df"],
                         save_path=f"{fig_dir}/next_season.png")
    except Exception as e:
        logger.warning(f"  Next-season plot failed (non-fatal): {e}")

    logger.info("Pipeline complete!")
    return weekly_df, model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILI Forecasting Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    setup_logging(args.log_level)
    main(args.config)
