"""
visualization.py — chart builders + the Tkinter embedding helper.

WHY THIS MODULE EXISTS (for learning):
Keeping "how to draw a chart" separate from "how to show it in a window"
keeps things testable: these functions build a matplotlib Figure and return
it; the GUI just asks for a figure and embeds it. If you ever swap Tkinter
for a web UI (e.g. Streamlit), you reuse every function in this file as-is.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def actual_vs_predicted_figure(test_dates, actual, predicted) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(test_dates, actual, label="Actual", marker="o", color="#1f77b4")
    ax.plot(test_dates, predicted, label="Predicted", marker="x", color="#d62728")
    ax.set_title("Actual vs. Predicted Water Supply (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Supply")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def forecast_figure(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["date"], history_df["water_supply"], label="Historical", color="#1f77b4")
    ax.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast 2026-2028", color="#2ca02c")
    ax.fill_between(
        forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"],
        color="#2ca02c", alpha=0.2, label="95% Confidence Interval",
    )
    ax.set_title("Water Supply Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Supply")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def residuals_figure(dates, residual_values) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0, color="gray", linewidth=1)
    ax.plot(dates, residual_values, marker="o", linestyle="none", color="#9467bd")
    ax.set_title("Forecast Residuals (Actual - Predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def embed_figure(figure: plt.Figure, parent_frame):
    """Embeds a matplotlib Figure into a Tkinter frame, replacing whatever
    was there before. Returns the canvas widget in case the caller needs it."""
    for child in parent_frame.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(figure, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas
