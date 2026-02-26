"""
Plotting functions for ILI forecasting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _save_or_show(fig, save_path: str = None):
    """Save figure and close, or show interactively."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_forecast(train: pd.DataFrame, test: pd.DataFrame,
                  forecast: np.ndarray, model_name: str = "SARIMAX",
                  ci_lower: np.ndarray = None, ci_upper: np.ndarray = None,
                  save_path: str = None):
    """Plot forecast vs actual with train context and 95% CI."""
    actual = np.expm1(test["log_ILI"]).values

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Full view
    axes[0].plot(train["week_start_date"], np.expm1(train["log_ILI"]),
                 color="steelblue", linewidth=0.8, label="Train")
    axes[0].plot(test["week_start_date"], actual,
                 color="black", linewidth=1.5, label="Actual")
    axes[0].plot(test["week_start_date"], forecast,
                 color="tomato", linewidth=1.5, linestyle="--", label="Forecast")
    axes[0].set_title(f"{model_name} — Full View", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("ILI Cases")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Test zoom with CI
    axes[1].plot(test["week_start_date"], actual,
                 color="black", linewidth=2, marker="o", markersize=3, label="Actual")
    axes[1].plot(test["week_start_date"], forecast,
                 color="tomato", linewidth=2, linestyle="--", marker="s",
                 markersize=3, label="Forecast")

    if ci_lower is not None and ci_upper is not None:
        axes[1].fill_between(test["week_start_date"],
                              ci_lower, ci_upper,
                              alpha=0.15, color="tomato", label="95% CI")
    else:
        # Fallback to ±20% band if no CI provided
        axes[1].fill_between(test["week_start_date"],
                              forecast * 0.8, forecast * 1.2,
                              alpha=0.15, color="tomato", label="±20% band")

    axes[1].set_title("Test Period", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("ILI Cases")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray,
                              model_name: str = "Model", save_path: str = None):
    """Scatter plot: actual vs predicted."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, edgecolors="black", s=40)
    max_val = max(actual.max(), predicted.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect")
    ax.set_title(f"{model_name} — Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual ILI Cases")
    ax.set_ylabel("Predicted ILI Cases")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_baselines_comparison(test: pd.DataFrame, models: dict,
                               save_path: str = None):
    """Plot all models against actual."""
    actual = np.expm1(test["log_ILI"]).values

    colors = {
        "Naive (last week)": "gray",
        "Seasonal Naive (ly)": "green",
        "Weekly Mean": "purple",
        "Weekly Median": "orange",
        "SARIMAX (rolling)": "tomato",
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(test["week_start_date"].values, actual,
            color="black", linewidth=2.5, label="Actual")

    for name, preds in models.items():
        color = colors.get(name, "gray")
        style = "--" if "SARIMAX" in name else ":"
        width = 2 if "SARIMAX" in name else 1
        alpha = 1.0 if "SARIMAX" in name else 0.6
        ax.plot(test["week_start_date"].values, preds,
                color=color, linewidth=width, linestyle=style,
                alpha=alpha, label=name)

    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("ILI Cases")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_residual_diagnostics(actual: np.ndarray, predicted: np.ndarray,
                               dates: pd.Series = None,
                               model_name: str = "SARIMAX",
                               save_path: str = None):
    """
    Residual diagnostic plots: residuals over time, histogram, ACF, PACF.

    Parameters
    ----------
    actual : np.ndarray
    predicted : np.ndarray
    dates : pd.Series (optional)
        Date index for the x-axis
    model_name : str
    save_path : str (optional)
    """
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Residuals over time
    x = dates if dates is not None else np.arange(len(residuals))
    axes[0, 0].plot(x, residuals, color="steelblue", linewidth=0.8)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals Over Time", fontweight="bold")
    axes[0, 0].set_ylabel("Residual (Actual - Predicted)")
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residual Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Frequency")

    # 3. ACF of residuals
    plot_acf(residuals, lags=40, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("ACF of Residuals", fontweight="bold")

    # 4. PACF of residuals
    plot_pacf(residuals, lags=40, ax=axes[1, 1], alpha=0.05, method="ywm")
    axes[1, 1].set_title("PACF of Residuals", fontweight="bold")

    fig.suptitle(f"{model_name} — Residual Diagnostics",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_season_panels(weekly_df: pd.DataFrame, save_path: str = None):
    """Plot ILI vs temperature per HMM-detected season."""
    season_data = weekly_df[weekly_df["season_label"].notna()].copy()
    seasons = sorted(season_data["season_label"].unique())

    ncols = 4
    nrows = (len(seasons) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for i, season in enumerate(seasons):
        ax = axes[i]
        s = season_data[season_data["season_label"] == season]

        ax.plot(s["week_start_date"], s["ILI_CASE"], color="steelblue", linewidth=1.5)
        ax.fill_between(s["week_start_date"], s["ILI_CASE"], alpha=0.15, color="steelblue")
        ax.set_ylabel("ILI", fontsize=8, color="steelblue")

        ax2 = ax.twinx()
        ax2.plot(s["week_start_date"], s["tmean_c"], color="tomato",
                 linewidth=1.5, linestyle="--")
        ax2.set_ylabel("°C", fontsize=8, color="tomato")

        ax.set_title(f"Season {int(season)}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(seasons), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("ILI vs Temperature per Season",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_hmm_seasonality(weekly_df: pd.DataFrame, save_path: str = None):
    """
    Two-panel figure showing HMM season detection across the full timeline.

    Top panel (3:1 height): ILI time series with season periods shaded
    Bottom panel: Season probability over time with 0.5 threshold
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    dates = weekly_df["week_start_date"]
    ili = weekly_df["ILI_CASE"]
    is_season = weekly_df["is_season"].values

    # --- Top panel: ILI time series with season shading ---
    axes[0].plot(dates, ili, color="steelblue", linewidth=0.9, label="ILI Cases")

    # Shade season periods
    season_data = weekly_df[weekly_df["season_label"].notna()]
    for season in sorted(season_data["season_label"].unique()):
        s = season_data[season_data["season_label"] == season]
        start = s["week_start_date"].iloc[0]
        end = s["week_start_date"].iloc[-1]
        axes[0].axvspan(start, end, alpha=0.2, color="tomato")

        # Annotate season label with week range above peak
        peak_idx = s["ILI_CASE"].idxmax()
        peak_date = s.loc[peak_idx, "week_start_date"]
        peak_val = s.loc[peak_idx, "ILI_CASE"]
        onset_week = start.isocalendar()[1]
        end_week = end.isocalendar()[1]
        peak_week = peak_date.isocalendar()[1]
        axes[0].annotate(
            f"{int(season)}\nW{onset_week}-W{end_week}\npeak W{peak_week}",
            xy=(peak_date, peak_val),
            xytext=(0, 18), textcoords="offset points",
            ha="center", fontsize=7, fontweight="bold", color="tomato",
        )

    axes[0].set_ylabel("ILI Cases")
    axes[0].set_title("HMM Flu Season Detection", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # --- Bottom panel: season probability ---
    axes[1].plot(dates, weekly_df["season_prob"], color="tomato", linewidth=0.9)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
    axes[1].fill_between(dates, weekly_df["season_prob"], alpha=0.15, color="tomato")
    axes[1].set_ylabel("Season Prob.")
    axes[1].set_xlabel("Date (week number shown)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Add week numbers to x-axis tick labels
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y\n(W%V)")
    )

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_next_season(weekly_df: pd.DataFrame, future_df: pd.DataFrame,
                     save_path: str = None):
    """
    Plot next-season forecast in three panels.

    Panel A: Full history + forecast with 95% CI
    Panel B: Recent 60 weeks zoom + forecast with peak annotation
    Panel C: Predicted season aligned by peak vs last 4 historical seasons
    """
    last_date = weekly_df["week_start_date"].max()
    predicted = future_df["predicted_ILI"].values
    ci_lower = future_df["ci_lower_95"].values
    ci_upper = future_df["ci_upper_95"].values
    peak_idx = int(np.argmax(predicted))

    # Compute a sensible y-axis cap from historical peaks + forecast
    hist_peak = weekly_df["ILI_CASE"].max()
    y_cap = max(hist_peak, predicted.max()) * 1.3
    ci_upper_clipped = np.minimum(ci_upper, y_cap)

    fig, axes = plt.subplots(3, 1, figsize=(18, 16))

    # --- A: Full timeline ---
    axes[0].plot(weekly_df["week_start_date"], weekly_df["ILI_CASE"],
                 color="steelblue", linewidth=1, label="Historical")
    axes[0].plot(future_df["week_start_date"], predicted,
                 color="tomato", linewidth=2.5, linestyle="--",
                 label="Forecast")
    axes[0].fill_between(future_df["week_start_date"], ci_lower, ci_upper_clipped,
                          alpha=0.15, color="tomato", label="95% CI")
    axes[0].axvline(last_date, color="gray", linestyle=":", alpha=0.7,
                    label="Forecast start")
    axes[0].set_title("ILI Cases: History + Next Season Forecast",
                       fontsize=15, fontweight="bold")
    axes[0].set_ylabel("ILI Cases")
    axes[0].set_ylim(0, y_cap)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # --- B: Zoom recent 60 weeks ---
    recent_start = last_date - pd.Timedelta(weeks=60)
    recent = weekly_df[weekly_df["week_start_date"] >= recent_start]

    axes[1].plot(recent["week_start_date"], recent["ILI_CASE"],
                 color="steelblue", linewidth=2, marker="o", markersize=3,
                 label="Actual")
    axes[1].plot(future_df["week_start_date"], predicted,
                 color="tomato", linewidth=2.5, linestyle="--", marker="s",
                 markersize=4, label="Forecast")
    axes[1].fill_between(future_df["week_start_date"], ci_lower, ci_upper_clipped,
                          alpha=0.12, color="tomato", label="95% CI")
    axes[1].axvline(last_date, color="gray", linestyle=":", alpha=0.7)

    # Mark peak
    peak_date = future_df["week_start_date"].iloc[peak_idx]
    peak_val = predicted[peak_idx]
    axes[1].scatter(peak_date, peak_val,
                    s=200, color="red", zorder=5,
                    edgecolors="black", linewidths=1.5)
    axes[1].annotate(
        f"Peak: {peak_val:,.0f}\n{peak_date.strftime('%d %b %Y')}",
        xy=(peak_date, peak_val),
        xytext=(15, 15), textcoords="offset points",
        fontsize=11, fontweight="bold", color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    axes[1].set_title("Recent + Forecast",
                       fontsize=15, fontweight="bold")
    axes[1].set_ylabel("ILI Cases")
    axes[1].set_ylim(0, y_cap)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    # --- C: Season overlay aligned by peak ---
    season_data = weekly_df[weekly_df["season_label"].notna()].copy()
    seasons_to_plot = sorted(season_data["season_label"].dropna().unique())[-4:]
    colors_hist = plt.cm.Blues(np.linspace(0.3, 0.9, len(seasons_to_plot)))

    for idx, season in enumerate(seasons_to_plot):
        s = season_data[season_data["season_label"] == season].copy()
        if len(s) == 0:
            continue
        peak_pos = s["ILI_CASE"].values.argmax()
        week_idx = np.arange(len(s)) - peak_pos
        axes[2].plot(week_idx, s["ILI_CASE"].values, color=colors_hist[idx],
                     linewidth=1.5, alpha=0.7, label=f"Season {int(season)}")

    pred_week_idx = np.arange(len(predicted)) - peak_idx
    axes[2].plot(pred_week_idx, predicted,
                 color="tomato", linewidth=3, linestyle="--",
                 label="Predicted next season")
    axes[2].fill_between(pred_week_idx, ci_lower, np.minimum(ci_upper, y_cap),
                          alpha=0.1, color="tomato")

    axes[2].set_xlabel("Weeks from Peak")
    axes[2].set_ylabel("ILI Cases")
    axes[2].set_title("Predicted vs Historical Seasons (Aligned by Peak)",
                       fontsize=15, fontweight="bold")
    axes[2].set_ylim(0, y_cap)
    axes[2].legend(fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].axvline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save_or_show(fig, save_path)
