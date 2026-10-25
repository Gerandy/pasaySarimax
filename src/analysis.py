"""
analysis.py — exploratory data analysis (EDA).

WHY THIS MODULE EXISTS (for learning):
Before trusting any forecast, we need to *understand* the data: is there a
trend? Does water supply rise and fall with the seasons? Are rainfall and
temperature actually related to water supply? This is standard practice in
any data science pipeline and is literally "Data Analysis" in your proposal
(section 2). Every function here returns a matplotlib Figure so the GUI can
embed it directly — analysis code never talks to Tkinter.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # safe default backend; GUI embeds figures via FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def historical_trend_figure(df: pd.DataFrame, column: str, title: str, ylabel: str) -> plt.Figure:
    """Simple line plot of one variable over time (e.g. historical water supply)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["date"], df[column], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def seasonal_decomposition_figure(df: pd.DataFrame, column: str = "water_supply", period: int = 12) -> plt.Figure:
    """Breaks a series into Trend + Seasonal + Residual components.

    This directly answers "Identification of seasonal patterns" from the
    proposal: the seasonal component shows the repeating yearly cycle, the
    trend shows the long-term direction, and the residual is "noise" left
    over once both are removed.
    """
    series = df.set_index("date")[column]
    result = seasonal_decompose(series, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual", linestyle="none", marker="o", markersize=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between water_supply, rainfall, temp_min, temp_max,
    and humidity (if present).

    A quick way to check: does more rainfall actually correlate with more
    water supply in this dataset? This justifies using these variables as
    SARIMAX "exogenous" (external) variables.
    """
    candidates = ["water_supply", "rainfall", "temp_min", "temp_max", "humidity"]
    columns = [c for c in candidates if c in df.columns]
    return df[columns].corr()


def correlation_heatmap_figure(df: pd.DataFrame) -> plt.Figure:
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig
