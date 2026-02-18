# ILI Forecasting Model — Israel

Weekly Influenza-Like Illness (ILI) case forecasting for Israel using WHO FluID surveillance data.

## Project Structure

```
ili_forecast/
├── config/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── raw/                     # Original downloaded files (never modified)
│   │   ├── VIW_FID_EPI.csv      # WHO FluID data
│   │   ├── Jewish_Israeli_holidays.csv
│   │   └── data_202512022119.csv # Temperature (IMS)
│   ├── processed/               # Pipeline outputs
│   │   └── weekly_df.parquet
│   └── external/                # Optional additional data (RSV, etc.)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py         # Load & clean raw sources
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py    # Feature engineering pipeline
│   │   ├── holidays.py          # Holiday processing
│   │   ├── school_calendar.py   # School session features
│   │   ├── temperature.py       # Temperature features
│   │   └── southern_hemisphere.py  # Australia regressors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sarimax_model.py     # SARIMAX training & forecasting
│   │   ├── baseline.py          # Baseline models
│   │   └── hmm_season.py        # HMM season detection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Evaluation metrics & comparison
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # All plotting functions
├── notebooks/
│   └── 01_exploration.ipynb     # EDA & experimentation
├── tests/
│   └── test_features.py
├── outputs/
│   ├── models/                  # Saved model objects
│   ├── figures/                 # Generated plots
│   └── reports/                 # Results summaries
├── main.py                      # Full pipeline runner
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run with custom config
python main.py --config config/config.yaml
```

## Model Performance (SARIMAX)

| Metric | Value |
|--------|-------|
| MAPE | 12.3% |
| R² | 0.898 |
| Directional Accuracy (in-season) | 88.0% |
| Within ±20% | 82.7% |

## Data Sources

- **ILI Cases**: WHO FluID (2011–2025)
- **Temperature**: Israel Meteorological Service, Tel Aviv station
- **Holidays**: Jewish/Israeli calendar
- **Southern Hemisphere**: WHO FluID — Australia
