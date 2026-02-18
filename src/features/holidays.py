"""
Israeli holiday processing for Prophet and feature engineering.
"""
import pandas as pd
import holidays as holidays_pkg


def build_holidays_df(
    csv_path: str,
    holidays_config: dict,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Build holidays DataFrame for Prophet.

    Combines CSV holidays + python holidays package,
    maps to consolidated groups, keeps first day per group per year.

    Returns
    -------
    pd.DataFrame with columns: holiday, ds, lower_window, upper_window
    """
    # Source 1: CSV
    csv_df = pd.read_csv(csv_path)
    csv_df = csv_df.iloc[:, :2]
    csv_df.columns = ["date", "holiday_name"]
    csv_df["ds"] = pd.to_datetime(csv_df["date"], errors="coerce")
    csv_df["holiday_raw"] = csv_df["holiday_name"].str.strip().str.lower()
    csv_df = csv_df[["ds", "holiday_raw"]].dropna()

    # Source 2: holidays package
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    il_holidays = holidays_pkg.Israel(years=range(start_year, end_year + 1))
    pkg_df = pd.DataFrame(
        [(pd.Timestamp(d), n) for d, n in il_holidays.items()],
        columns=["ds", "holiday_raw"]
    )
    pkg_df["holiday_raw"] = pkg_df["holiday_raw"].str.strip().str.lower()

    # Combine and deduplicate
    all_holidays = (
        pd.concat([csv_df, pkg_df], ignore_index=True)
        .dropna()
        .drop_duplicates(subset=["ds"])
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Map to groups
    raw_to_group = {}
    for group, names in holidays_config["groups"].items():
        for name in names:
            raw_to_group[name] = group

    all_holidays["holiday"] = all_holidays["holiday_raw"].map(raw_to_group).fillna("other")

    # Keep first day per group per year
    all_holidays["year"] = all_holidays["ds"].dt.year
    holidays_combined = (
        all_holidays
        .sort_values("ds")
        .groupby(["year", "holiday"], as_index=False)
        .first()
        [["holiday", "ds"]]
    )

    # Effect windows
    long = holidays_config["long_holidays"]
    short_win = holidays_config["short_window"]
    long_win = holidays_config["long_window"]

    holidays_combined["lower_window"] = 0
    holidays_combined["upper_window"] = short_win
    holidays_combined.loc[
        holidays_combined["holiday"].isin(long), "upper_window"
    ] = long_win

    # Filter to date range
    holidays_combined = holidays_combined[
        (holidays_combined["ds"] >= start_date) &
        (holidays_combined["ds"] <= end_date)
    ].reset_index(drop=True)

    return holidays_combined
