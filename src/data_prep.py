"""
data_prep.py — loading, validating, and cleaning the OASIS dataset.

WHY THIS MODULE EXISTS (for learning):
Real-world datasets are messy (missing rows, wrong types, unsorted dates).
A forecasting model like SARIMAX is sensitive to gaps in a time series, so we
isolate all "make the data trustworthy" logic here, separate from analysis
and modeling. This separation (data -> analysis -> model -> evaluation -> UI)
is a common pattern: each module has ONE job, which makes the code easier to
test, debug, and explain in a thesis defense.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["date", "water_supply", "rainfall", "temp_min", "temp_max"]
OPTIONAL_COLUMNS = ["humidity"]
NUMERIC_COLUMNS = ["water_supply", "rainfall", "temp_min", "temp_max"]


class DataValidationError(Exception):
    """Raised when the loaded file doesn't match the expected OASIS schema."""


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Accepts .csv, .xlsx, .xls. Raises DataValidationError if required
    columns are missing so the GUI can show a clear message instead of
    crashing.
    """
    if filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise DataValidationError(
            f"{bad_rows} row(s) have an unparseable date. "
            "Expected a date-like column, e.g. YYYY-MM-DD."
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df


def present_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Required numeric columns plus any optional ones (e.g. humidity) that
    are actually present in this dataset."""
    return NUMERIC_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]


def missing_data_report(df: pd.DataFrame) -> dict:
    """Return a dict of column -> count of missing values. Used by the GUI
    to show the client/student what will be cleaned and why."""
    columns = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
    return {col: int(df[col].isna().sum()) for col in columns}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps using time-based linear interpolation.

    Why interpolation instead of just dropping rows? SARIMAX needs an evenly
    spaced time index (no missing months). Interpolation estimates a
    missing month from its neighbors, which is standard practice for short
    gaps in environmental/utility time series.
    """
    df = df.copy()
    numeric_cols = present_numeric_columns(df)
    df = df.set_index("date")
    df = df.asfreq("MS")  # ensure every month is present (Month Start freq)
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").bfill().ffill()
    df = df.reset_index()
    return df


def load_and_clean(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper used by the GUI: load, report missing data, clean."""
    raw = load_data(filepath)
    report = missing_data_report(raw)
    cleaned = clean_data(raw)
    return cleaned, report
