"""
HMM-based flu season detection.
"""
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# Seasons shorter than this many weeks are considered noise
MIN_SEASON_WEEKS = 4


def fit_hmm_seasons(weekly_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, GaussianHMM]:
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
    logger.info(
        f"HMM means: state 0={means[0]:,.0f}, state 1={means[1]:,.0f} "
        f"-> season state = {season_state}"
    )

    weekly_df["hmm_state"] = states
    weekly_df["season_prob"] = probs[:, season_state]
    weekly_df["is_season"] = (states == season_state).astype(int)

    # Filter out very short seasons (noise)
    weekly_df = _filter_short_seasons(weekly_df, min_weeks=MIN_SEASON_WEEKS)

    # Label individual seasons
    weekly_df = _label_seasons(weekly_df)

    return weekly_df, model


def _filter_short_seasons(weekly_df: pd.DataFrame, min_weeks: int = 4) -> pd.DataFrame:
    """Remove season blocks shorter than min_weeks (likely noise)."""
    is_s = weekly_df["is_season"].values.copy()
    blocks = _find_blocks(is_s, target=1)

    n_removed = 0
    for start, end in blocks:
        block_len = end - start + 1
        if block_len < min_weeks:
            is_s[start:end + 1] = 0
            n_removed += 1

    if n_removed > 0:
        logger.info(f"Removed {n_removed} short season blocks (< {min_weeks} weeks)")

    weekly_df["is_season"] = is_s
    return weekly_df


def _find_blocks(arr: np.ndarray, target: int = 1) -> list:
    """Find contiguous blocks of a target value. Returns list of (start, end) pairs."""
    blocks = []
    in_block = False
    start = 0

    for i in range(len(arr)):
        if arr[i] == target and not in_block:
            start = i
            in_block = True
        elif arr[i] != target and in_block:
            blocks.append((start, i - 1))
            in_block = False

    if in_block:
        blocks.append((start, len(arr) - 1))

    return blocks


def _label_seasons(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each continuous season block with a unique year label.
    Based on the January/February within the season.
    """
    weekly_df = weekly_df.reset_index(drop=True)
    weekly_df["season_label"] = np.nan

    is_s = weekly_df["is_season"].values
    blocks = _find_blocks(is_s, target=1)

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
