# ILI Forecasting Pipeline — Israel

Weekly Influenza-Like Illness (ILI) case forecasting for Israel using WHO FluID surveillance data and SARIMAX time series modeling.

## Overview

This project builds an end-to-end forecasting pipeline that predicts weekly ILI case counts in Israel. The core insight is that flu in the Southern Hemisphere (Australia) peaks ~6 months before Israel's season, providing a leading indicator. The model combines this with local temperature data, school calendar patterns, and COVID-era adjustments to forecast seasonal dynamics.

**Key result:** The model predicts the 2025–26 flu season will peak at ~9,000 cases around January 11, 2026 (95% CI: 5,220–15,686), suggesting a moderate season compared to the prior two years.

## Model Performance

**SARIMAX(3,1,0)×(1,0,1,52) — Rolling one-step-ahead evaluation on 128 test weeks:**

| Metric | Overall | In-Season | Off-Season |
|--------|---------|-----------|------------|
| MAE | 314 | 721 | 45 |
| RMSE | 727 | 1,149 | 65 |
| MAPE | 21.6% | 21.5% | 21.7% |
| R² | 0.923 | — | — |
| Directional Accuracy | 59.8% | 86.0% | 39.5% |

**Baseline comparison:**

| Model | MAE | MAPE | R² |
|-------|-----|------|-----|
| **SARIMAX (rolling)** | **314** | **21.6%** | **0.923** |
| Naive (last week) | 392 | 26.1% | 0.906 |
| Seasonal Naive (last year) | 941 | 62.9% | 0.441 |
| Weekly Mean | 906 | 55.8% | 0.460 |
| Weekly Median | 927 | 60.5% | 0.451 |

## Features

| Feature | Source | Description |
|---------|--------|-------------|
| `tmean_c` | Israel Meteorological Service | Weekly mean temperature (°C) |
| `tmean_lag1`, `tmean_lag2` | IMS | 1- and 2-week lagged temperature |
| `AUS_ILI_LAG1` | WHO FluID — Australia | Australian ILI cases lagged 1 week |
| `school_in_session` | Israeli school calendar | Binary: is school in session |
| `is_covid`, `post_covid` | Calendar-based | COVID period flags (2020–2022) |

All features are known at prediction time — no target leakage.

## Pipeline Steps

```
[1/7] Build features      → Merge ILI, temperature, holidays, Australia data
[2/7] HMM season detection → 2-state Hidden Markov Model identifies flu seasons
[3/7] Train/test split     → Temporal split at 2023-06-01 (648 train / 128 test)
[4/7] Baseline models      → Naive, seasonal naive, weekly mean/median
[5/7] Fit SARIMAX          → SARIMAX(3,1,0)×(1,0,1,52) with exogenous features
[6/7] Rolling forecast     → One-step-ahead with Kalman filter state updates
[7/7] Next-season forecast → 35-week forward prediction with 95% CI
```

## Project Structure

```
ili_forecast/
├── config/
│   └── config.yaml                 # All hyperparameters and settings
├── data/
│   ├── raw/                        # Original downloaded files (never modified)
│   │   ├── VIW_FID_EPI.csv         # WHO FluID data
│   │   ├── Jewish_Israeli_holidays.csv
│   │   └── data_202512022119.csv   # Temperature (IMS)
│   └── processed/
│       └── weekly_df.parquet       # Feature matrix
├── src/
│   ├── data/
│   │   └── load_data.py            # Load & validate raw sources
│   ├── features/
│   │   ├── build_features.py       # Feature engineering & train/test split
│   │   ├── holidays.py             # Holiday processing
│   │   ├── school_calendar.py      # School session features
│   │   ├── temperature.py          # Temperature features
│   │   └── southern_hemisphere.py  # Australia ILI regressors
│   ├── models/
│   │   ├── sarimax_model.py        # SARIMAX training, rolling forecast & CI
│   │   ├── baseline.py             # Baseline models
│   │   ├── hmm_season.py           # HMM season detection
│   │   └── next_season_prediction.py # Forward prediction with exog projection
│   ├── evaluation/
│   │   └── metrics.py              # MAE, RMSE, MAPE, SMAPE, band accuracy
│   └── visualization/
│       └── plots.py                # Forecast, scatter, residual, comparison plots
├── tests/
│   └── test_features.py            # 25+ tests for features, metrics, models
├── outputs/
│   ├── models/                     # sarimax_best.joblib
│   ├── figures/                    # All generated plots
│   ├── model_comparison.csv
│   ├── sarimax_evaluation.csv
│   └── next_season_forecast.csv
├── notebooks/
│   └── 01_exploration.ipynb
├── main.py                         # Full pipeline runner
└── pyproject.toml
```

## Quick Start

```bash
# Install with Poetry
poetry install

# Run full pipeline
poetry run python main.py

# Run with custom config / log level
poetry run python main.py --config config/config.yaml --log-level DEBUG

# Run tests
poetry run pytest tests/
```

**Requirements:** Python 3.10–3.12

## Outputs

The pipeline generates:

- **`outputs/figures/sarimax_forecast.png`** — Forecast vs actual with 95% confidence interval
- **`outputs/figures/sarimax_scatter.png`** — Actual vs predicted scatter plot
- **`outputs/figures/sarimax_residuals.png`** — Residual diagnostics (time series, histogram, ACF, PACF)
- **`outputs/figures/model_comparison.png`** — SARIMAX vs all baselines
- **`outputs/figures/next_season.png`** — 2025–26 season forecast with 95% CI
- **`outputs/next_season_forecast.csv`** — Weekly predictions with confidence bounds
- **`outputs/model_comparison.csv`** — All models side by side
- **`outputs/sarimax_evaluation.csv`** — Detailed evaluation (overall, in-season, off-season)

## Data Sources

- **ILI Cases**: [WHO FluID](https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/fluid) — Israel & Australia (2009–2025)
- **Temperature**: Israel Meteorological Service — Tel Aviv station
- **Holidays**: Jewish/Israeli calendar
- **School Calendar**: Israeli Ministry of Education schedule

## Technical Notes

- Model is fitted in log-space (`log1p(ILI_cases)`) for variance stabilization; all outputs are back-transformed
- Rolling forecast uses `append(refit=False)` to update the Kalman filter state without refitting — 128 steps in ~20 seconds
- HMM-based season detection (2-state Gaussian) replaces manual threshold definitions
- Next-season prediction uses historical medians by epi-week for temperature and Australia features, and exact computation for calendar features
