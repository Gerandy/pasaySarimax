"""
model.py — SARIMAX training, testing, and forecasting.

WHY THIS MODULE EXISTS (for learning):
SARIMAX = Seasonal AutoRegressive Integrated Moving Average with eXogenous
regressors. In plain terms:
  - AR (p): predicts using past values of the series itself.
  - I  (d): "differencing" — how many times we subtract consecutive values
            to make the series stationary (remove trend).
  - MA (q): predicts using past forecast errors.
  - Seasonal (P, D, Q, s): same idea as above but for a repeating seasonal
            cycle. s=12 means "yearly seasonality" for monthly data.
  - eXogenous (rainfall, temperature): extra variables that help explain
            water_supply, fed in alongside the time series itself.

This module wraps statsmodels' SARIMAX so the GUI only needs to call a
couple of simple functions instead of knowing statsmodels' API.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 12)


@dataclass
class TrainTestSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def train_test_split_by_ratio(df: pd.DataFrame, test_ratio: float = 0.2) -> TrainTestSplit:
    """Chronological split (NOT random) — time series must keep order so we
    test on the most recent, unseen period, simulating a real forecast."""
    n_test = max(1, int(len(df) * test_ratio))
    train = df.iloc[: len(df) - n_test].reset_index(drop=True)
    test = df.iloc[len(df) - n_test :].reset_index(drop=True)
    return TrainTestSplit(train=train, test=test)


def _exog(df: pd.DataFrame, exog_columns: list[str]) -> pd.DataFrame | None:
    if not exog_columns:
        return None
    return df[exog_columns]


def fit_sarimax(
    train: pd.DataFrame,
    exog_columns: list[str] | None = None,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
):
    """Fit a SARIMAX model on the training set.

    exog_columns: e.g. ["rainfall", "temperature"] to use them as external
    regressors, or [] / None to run a pure time-series model.
    """
    exog_columns = exog_columns or []
    endog = train["water_supply"]
    exog = _exog(train, exog_columns)
    model = SARIMAX(
        endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def evaluate_on_test(fitted_model, test: pd.DataFrame, exog_columns: list[str] | None = None) -> pd.Series:
    """Predict over the held-out test period and return predictions aligned
    with test's dates, so actual vs. predicted can be plotted/compared."""
    exog_columns = exog_columns or []
    exog = _exog(test, exog_columns)
    steps = len(test)
    pred = fitted_model.get_forecast(steps=steps, exog=exog)
    predicted = pred.predicted_mean
    predicted.index = test["date"].values
    return predicted


def forecast_future(
    fitted_model,
    full_df: pd.DataFrame,
    periods: int,
    exog_columns: list[str] | None = None,
    future_exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast `periods` months beyond the last date in full_df (e.g. 36
    months to cover 2026-2028).

    If exog_columns are used, future_exog must supply rainfall/temperature
    projections for those future months (e.g. historical monthly averages),
    since SARIMAX needs them to produce the forecast.
    """
    exog_columns = exog_columns or []
    exog = future_exog[exog_columns] if (exog_columns and future_exog is not None) else None
    pred = fitted_model.get_forecast(steps=periods, exog=exog)
    mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=0.05)

    last_date = full_df["date"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="MS")[1:]

    result = pd.DataFrame({
        "date": future_dates,
        "forecast": mean.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values,
    })
    return result


def naive_future_exog(df: pd.DataFrame, periods: int, exog_columns: list[str]) -> pd.DataFrame:
    """Builds a simple future rainfall/temperature projection by repeating
    each month's historical average (seasonal climatology). This is a
    reasonable placeholder until real projected data is supplied."""
    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="MS")[1:]
    monthly_avg = df.set_index("date")[exog_columns].groupby(lambda d: d.month).mean()
    rows = [monthly_avg.loc[d.month] for d in future_dates]
    future_exog = pd.DataFrame(rows)
    future_exog.index = future_dates
    future_exog = future_exog.reset_index(drop=True)
    return future_exog
