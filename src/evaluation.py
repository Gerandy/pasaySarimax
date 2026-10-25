"""
evaluation.py — forecast accuracy metrics.

WHY THIS MODULE EXISTS (for learning):
A forecast is only useful if we know how wrong it tends to be. The proposal
specifically asks for MAE and RMSE (section 2, "Model Evaluation"):

  MAE  (Mean Absolute Error): average size of the error, in the same units
       as water_supply. Easy to explain to a non-technical client:
       "on average, the forecast is off by X units."

  RMSE (Root Mean Squared Error): similar, but squares errors before
       averaging, so large mistakes are penalized more. Useful for
       showing whether the model has occasional big misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluation_summary(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "MAE": round(mean_absolute_error(actual, predicted), 3),
        "RMSE": round(root_mean_squared_error(actual, predicted), 3),
    }


def residuals(actual: pd.Series, predicted: pd.Series) -> np.ndarray:
    """Errors over time (actual - predicted). Plotting these helps spot
    whether the model systematically over- or under-predicts in certain
    periods (e.g. always too low during the wet season)."""
    return np.asarray(actual) - np.asarray(predicted)
