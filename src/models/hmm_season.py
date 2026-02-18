"""
HMM-based flu season detection.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm_seasons(weekly_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fit 2-state Gaussian HMM to detect flu seasons.

    State 0: off-season (low ILI)
    State 1: season (high ILI)

    Adds columns: hmm_state, season_prob, is_season, season_label
    """
    hmm_cfg = config["hmm"]

    X = weekly_df["ILI_CASE"].values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=hmm_cfg["n_components"],
        covariance_type=hmm_cfg["covariance_type"],
        n_iter=hmm_cfg["n_iter"],
        random_state=hmm_cfg["random_state"]
    )
    model.fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)

    # Identify which state is "season" (higher mean)
    means = model.means_.flatten()
    season_state = np.argmax(means)

    weekly_df["hmm_state"] = states
    weekly_df["season_prob"] = probs[:, season_state]
    weekly_df["is_season"] = (states == season_state).astype(int)

    # Label individual seasons
    weekly_df = _label_seasons(weekly_df)

    return weekly_df


def _label_seasons(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each continuous season block with a unique year label.
    Based on the January/February within the season.
    """
    weekly_df = weekly_df.reset_index(drop=True)
    weekly_df["season_label"] = np.nan

    # Build list of (start, end) index pairs for each season block
    is_s = weekly_df["is_season"].values
    blocks = []
    in_block = False
    start = 0

    for i in range(len(is_s)):
        if is_s[i] == 1 and not in_block:
            start = i
            in_block = True
        elif is_s[i] == 0 and in_block:
            blocks.append((start, i - 1))
            in_block = False
    # Handle block that extends to the end
    if in_block:
        blocks.append((start, len(is_s) - 1))

    used_labels = set()

    for entry_idx, exit_idx in blocks:
        season_block = weekly_df.iloc[entry_idx:exit_idx + 1]
        jan_weeks = season_block[season_block["week_start_date"].dt.month.isin([1, 2])]

        if len(jan_weeks) > 0:
            label = int(jan_weeks["week_start_date"].dt.year.mode().iloc[0])
        else:
            label = int(season_block["week_start_date"].dt.year.iloc[0]) + 1

        # Collision detection
        while label in used_labels:
            label += 1
        used_labels.add(label)

        weekly_df.loc[entry_idx:exit_idx, "season_label"] = label

    return weekly_df
