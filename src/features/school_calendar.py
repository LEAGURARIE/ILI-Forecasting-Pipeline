"""
Israeli school calendar features.
"""
import pandas as pd


def is_school_in_session(date, config: dict) -> int:
    """
    Israeli school calendar:
    Summer break: ~June 20 – Aug 31
    Rest of year: in session (holidays handled separately)
    """
    month = date.month
    day = date.day

    start_m = config["summer_start_month"]
    start_d = config["summer_start_day"]
    end_m = config["summer_end_month"]
    end_d = config["summer_end_day"]

    # Fully inside summer months
    if start_m < month < end_m:
        return 0
    # Start month: on or after start day
    if month == start_m and day >= start_d:
        return 0
    # End month: on or before end day
    if month == end_m and day <= end_d:
        return 0
    return 1


def weeks_since_school_start(date, config: dict) -> float:
    """
    Weeks since Sep 1 — captures infection buildup in classrooms.
    School starts → kids mix → viruses spread → ILI rises after 8-12 weeks.
    """
    year = date.year
    school_month = config["school_start_month"]
    school_day = config["school_start_day"]
    school_start = pd.Timestamp(f"{year}-{school_month:02d}-{school_day:02d}")

    if date < school_start:
        school_start = pd.Timestamp(f"{year - 1}-{school_month:02d}-{school_day:02d}")

    diff = (date - school_start).days / 7
    return max(0, diff)


def add_school_features(weekly_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add school_in_session and weeks_since_school to weekly_df."""
    weekly_df["school_in_session"] = weekly_df["week_start_date"].apply(
        lambda d: is_school_in_session(d, config)
    )
    weekly_df["weeks_since_school"] = weekly_df["week_start_date"].apply(
        lambda d: weeks_since_school_start(d, config)
    )
    return weekly_df
