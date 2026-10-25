"""
gui.py — Tkinter desktop application tying every module together.

WHY THIS MODULE EXISTS (for learning):
This is the only file that imports `tkinter`. Every other module (data_prep,
analysis, model, evaluation, visualization) works with plain DataFrames and
matplotlib Figures and could be reused in a notebook, a web app, or unit
tests without ever touching a GUI framework. The GUI's job is just:
  1. ask the user for input (files, parameters, button clicks)
  2. call the appropriate module function
  3. display the result (chart, table, text)

The app is organized as four tabs (ttk.Notebook), mirroring the proposal's
workflow: Data -> Analysis -> Forecast -> Evaluation.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from . import data_prep, analysis, model, evaluation, visualization

EXOG_OPTIONS = ["rainfall", "temp_min", "temp_max", "humidity"]


class OasisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OASIS — Pasonanca Watershed Water Supply Forecasting")
        self.root.geometry("1000x700")

        # shared state across tabs
        self.df: pd.DataFrame | None = None
        self.fitted_model = None
        self.split: model.TrainTestSplit | None = None
        self.predicted_test: pd.Series | None = None
        self.forecast_df: pd.DataFrame | None = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.data_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.forecast_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text="1. Data")
        self.notebook.add(self.analysis_tab, text="2. Analysis")
        self.notebook.add(self.forecast_tab, text="3. Forecast")
        self.notebook.add(self.evaluation_tab, text="4. Evaluation")

        self._build_data_tab()
        self._build_analysis_tab()
        self._build_forecast_tab()
        self._build_evaluation_tab()

    # ------------------------------------------------------------------ #
    # Tab 1: Data
    # ------------------------------------------------------------------ #
    def _build_data_tab(self):
        top = ttk.Frame(self.data_tab, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Load CSV / Excel File", command=self._on_load_file).pack(side="left")
        self.file_label = ttk.Label(top, text="No file loaded.")
        self.file_label.pack(side="left", padx=10)

        self.missing_label = ttk.Label(self.data_tab, text="", padding=10, foreground="#555")
        self.missing_label.pack(fill="x")

        columns = data_prep.REQUIRED_COLUMNS  # rebuilt dynamically once real data (with optional columns) loads
        self.preview_tree = ttk.Treeview(self.data_tab, columns=columns, show="headings", height=15)
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=140, anchor="center")
        self.preview_tree.pack(fill="both", expand=True, padx=10, pady=10)

    def _on_load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select historical data file",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")],
        )
        if not filepath:
            return
        try:
            cleaned, report = data_prep.load_and_clean(filepath)
        except data_prep.DataValidationError as exc:
            messagebox.showerror("Invalid data file", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surface any unexpected parse error to the user
            messagebox.showerror("Failed to load file", str(exc))
            return

        self.df = cleaned
        self.file_label.config(text=filepath)
        missing_text = ", ".join(f"{col}: {count}" for col, count in report.items())
        self.missing_label.config(text=f"Missing values found (now interpolated) -> {missing_text}")
        self._refresh_preview()
        self._refresh_exog_availability()
        messagebox.showinfo("Data loaded", f"Loaded and cleaned {len(cleaned)} monthly rows.")

    def _refresh_preview(self):
        columns = list(self.df.columns)  # includes humidity automatically if present
        self.preview_tree.config(columns=columns)
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=120, anchor="center")
        self.preview_tree.delete(*self.preview_tree.get_children())
        for _, row in self.df.head(50).iterrows():
            values = [
                row[col].strftime("%Y-%m-%d") if col == "date" else row[col]
                for col in columns
            ]
            self.preview_tree.insert("", "end", values=values)

    # ------------------------------------------------------------------ #
    # Tab 2: Analysis
    # ------------------------------------------------------------------ #
    def _build_analysis_tab(self):
        controls = ttk.Frame(self.analysis_tab, padding=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Historical Water Supply", command=self._show_trend_water_supply).pack(side="left", padx=4)
        ttk.Button(controls, text="Rainfall Trend", command=self._show_trend_rainfall).pack(side="left", padx=4)
        ttk.Button(controls, text="Temp Min Trend", command=self._show_trend_temp_min).pack(side="left", padx=4)
        ttk.Button(controls, text="Temp Max Trend", command=self._show_trend_temp_max).pack(side="left", padx=4)
        ttk.Button(controls, text="Humidity Trend", command=self._show_trend_humidity).pack(side="left", padx=4)
        ttk.Button(controls, text="Seasonal Decomposition", command=self._show_seasonal_decomposition).pack(side="left", padx=4)
        ttk.Button(controls, text="Correlation Heatmap", command=self._show_correlation).pack(side="left", padx=4)

        self.analysis_chart_frame = ttk.Frame(self.analysis_tab)
        self.analysis_chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _require_data(self) -> bool:
        if self.df is None:
            messagebox.showwarning("No data", "Please load a data file in the Data tab first.")
            return False
        return True

    def _show_trend_water_supply(self):
        if not self._require_data():
            return
        fig = analysis.historical_trend_figure(self.df, "water_supply", "Historical Water Supply", "Water Supply")
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_trend_rainfall(self):
        if not self._require_data():
            return
        fig = analysis.historical_trend_figure(self.df, "rainfall", "Historical Rainfall", "Rainfall (mm)")
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_trend_temp_min(self):
        if not self._require_data():
            return
        fig = analysis.historical_trend_figure(self.df, "temp_min", "Historical Min Temperature", "Temp Min (°C)")
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_trend_temp_max(self):
        if not self._require_data():
            return
        fig = analysis.historical_trend_figure(self.df, "temp_max", "Historical Max Temperature", "Temp Max (°C)")
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_trend_humidity(self):
        if not self._require_data():
            return
        if "humidity" not in self.df.columns:
            messagebox.showwarning("No humidity data", "This dataset doesn't have a 'humidity' column.")
            return
        fig = analysis.historical_trend_figure(self.df, "humidity", "Historical Humidity", "Humidity (%)")
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_seasonal_decomposition(self):
        if not self._require_data():
            return
        if len(self.df) < 24:
            messagebox.showwarning("Not enough data", "Seasonal decomposition needs at least 2 years of monthly data.")
            return
        fig = analysis.seasonal_decomposition_figure(self.df)
        visualization.embed_figure(fig, self.analysis_chart_frame)

    def _show_correlation(self):
        if not self._require_data():
            return
        fig = analysis.correlation_heatmap_figure(self.df)
        visualization.embed_figure(fig, self.analysis_chart_frame)

    # ------------------------------------------------------------------ #
    # Tab 3: Forecast
    # ------------------------------------------------------------------ #
    def _build_forecast_tab(self):
        params = ttk.LabelFrame(self.forecast_tab, text="SARIMAX Parameters", padding=10)
        params.pack(fill="x", padx=10, pady=10)

        # order (p, d, q)
        self.p_var = tk.IntVar(value=1)
        self.d_var = tk.IntVar(value=1)
        self.q_var = tk.IntVar(value=1)
        # seasonal order (P, D, Q, s)
        self.sp_var = tk.IntVar(value=1)
        self.sd_var = tk.IntVar(value=1)
        self.sq_var = tk.IntVar(value=1)
        self.s_var = tk.IntVar(value=12)
        self.test_ratio_var = tk.DoubleVar(value=0.2)
        self.horizon_var = tk.IntVar(value=36)  # 36 months = 2026-2028

        def spin(parent, var, frm, to, label):
            ttk.Label(parent, text=label).pack(side="left")
            ttk.Spinbox(parent, textvariable=var, from_=frm, to=to, width=4).pack(side="left", padx=(2, 10))

        order_row = ttk.Frame(params)
        order_row.pack(fill="x", pady=2)
        ttk.Label(order_row, text="Order (p, d, q):").pack(side="left", padx=(0, 6))
        spin(order_row, self.p_var, 0, 5, "p")
        spin(order_row, self.d_var, 0, 2, "d")
        spin(order_row, self.q_var, 0, 5, "q")

        seasonal_row = ttk.Frame(params)
        seasonal_row.pack(fill="x", pady=2)
        ttk.Label(seasonal_row, text="Seasonal (P, D, Q, s):").pack(side="left", padx=(0, 6))
        spin(seasonal_row, self.sp_var, 0, 5, "P")
        spin(seasonal_row, self.sd_var, 0, 2, "D")
        spin(seasonal_row, self.sq_var, 0, 5, "Q")
        spin(seasonal_row, self.s_var, 1, 24, "s")

        exog_row = ttk.Frame(params)
        exog_row.pack(fill="x", pady=2)
        ttk.Label(exog_row, text="External variables:").pack(side="left", padx=(0, 6))
        self.exog_vars = {name: tk.BooleanVar(value=(name != "humidity")) for name in EXOG_OPTIONS}
        self.exog_checkbuttons = {}
        for name in EXOG_OPTIONS:
            cb = ttk.Checkbutton(exog_row, text=name, variable=self.exog_vars[name])
            cb.pack(side="left", padx=4)
            self.exog_checkbuttons[name] = cb
        # humidity is optional and starts disabled until a dataset that has it is loaded
        self.exog_checkbuttons["humidity"].state(["disabled"])

        misc_row = ttk.Frame(params)
        misc_row.pack(fill="x", pady=2)
        ttk.Label(misc_row, text="Test set size (fraction):").pack(side="left")
        ttk.Spinbox(misc_row, textvariable=self.test_ratio_var, from_=0.1, to=0.4, increment=0.05, width=5).pack(side="left", padx=(2, 20))
        ttk.Label(misc_row, text="Forecast horizon (months):").pack(side="left")
        ttk.Spinbox(misc_row, textvariable=self.horizon_var, from_=1, to=60, width=5).pack(side="left", padx=2)

        buttons = ttk.Frame(self.forecast_tab, padding=(10, 0))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="1. Train & Test Model", command=self._on_train_and_test).pack(side="left", padx=4)
        ttk.Button(buttons, text="2. Generate 2026-2028 Forecast", command=self._on_generate_forecast).pack(side="left", padx=4)
        ttk.Button(buttons, text="Export Forecast to CSV", command=self._on_export_forecast).pack(side="left", padx=4)

        body = ttk.Frame(self.forecast_tab)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        self.forecast_chart_frame = ttk.Frame(body)
        self.forecast_chart_frame.pack(side="left", fill="both", expand=True)

        table_frame = ttk.Frame(body)
        table_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(table_frame, text="Forecast Table").pack(anchor="w")
        self.forecast_tree = ttk.Treeview(
            table_frame, columns=["date", "forecast", "lower_ci", "upper_ci"], show="headings", height=20
        )
        for col in ["date", "forecast", "lower_ci", "upper_ci"]:
            self.forecast_tree.heading(col, text=col)
            self.forecast_tree.column(col, width=100, anchor="center")
        self.forecast_tree.pack(fill="both", expand=True)

    def _selected_exog_columns(self) -> list[str]:
        return [name for name, var in self.exog_vars.items() if var.get()]

    def _refresh_exog_availability(self):
        """Enable/disable the 'humidity' checkbox depending on whether the
        loaded dataset actually has that optional column."""
        has_humidity = self.df is not None and "humidity" in self.df.columns
        cb = self.exog_checkbuttons["humidity"]
        if has_humidity:
            cb.state(["!disabled"])
            self.exog_vars["humidity"].set(True)
        else:
            cb.state(["disabled"])
            self.exog_vars["humidity"].set(False)

    def _get_order_params(self):
        order = (self.p_var.get(), self.d_var.get(), self.q_var.get())
        seasonal_order = (self.sp_var.get(), self.sd_var.get(), self.sq_var.get(), self.s_var.get())
        return order, seasonal_order

    def _on_train_and_test(self):
        if not self._require_data():
            return
        if len(self.df) < 24:
            messagebox.showwarning("Not enough data", "Need at least 24 months of data to train/test SARIMAX.")
            return
        order, seasonal_order = self._get_order_params()
        exog_columns = self._selected_exog_columns()

        self.split = model.train_test_split_by_ratio(self.df, self.test_ratio_var.get())
        try:
            fitted = model.fit_sarimax(self.split.train, exog_columns, order, seasonal_order)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training failed", f"SARIMAX failed to fit: {exc}")
            return

        self.fitted_model = fitted
        predicted = model.evaluate_on_test(fitted, self.split.test, exog_columns)
        self.predicted_test = predicted

        fig = visualization.actual_vs_predicted_figure(
            self.split.test["date"], self.split.test["water_supply"], predicted.values
        )
        visualization.embed_figure(fig, self.forecast_chart_frame)

        summary = evaluation.evaluation_summary(self.split.test["water_supply"], predicted.values)
        self._update_evaluation(summary, self.split.test["date"], self.split.test["water_supply"], predicted.values)

        messagebox.showinfo("Model trained", f"Trained on {len(self.split.train)} months, tested on {len(self.split.test)} months.\nSee the Evaluation tab for MAE/RMSE.")

    def _on_generate_forecast(self):
        if not self._require_data():
            return
        order, seasonal_order = self._get_order_params()
        exog_columns = self._selected_exog_columns()
        horizon = self.horizon_var.get()

        try:
            fitted_full = model.fit_sarimax(self.df, exog_columns, order, seasonal_order)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Forecast failed", f"SARIMAX failed to fit on full data: {exc}")
            return

        future_exog = None
        if exog_columns:
            future_exog = model.naive_future_exog(self.df, horizon, exog_columns)

        forecast_df = model.forecast_future(fitted_full, self.df, horizon, exog_columns, future_exog)
        self.forecast_df = forecast_df

        fig = visualization.forecast_figure(self.df, forecast_df)
        visualization.embed_figure(fig, self.forecast_chart_frame)

        self.forecast_tree.delete(*self.forecast_tree.get_children())
        for _, row in forecast_df.iterrows():
            self.forecast_tree.insert("", "end", values=[
                row["date"].strftime("%Y-%m-%d"),
                round(row["forecast"], 2),
                round(row["lower_ci"], 2),
                round(row["upper_ci"], 2),
            ])

    def _on_export_forecast(self):
        if self.forecast_df is None:
            messagebox.showwarning("Nothing to export", "Generate a forecast first.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save forecast as", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")], initialfile="oasis_forecast_2026_2028.csv",
        )
        if not filepath:
            return
        self.forecast_df.to_csv(filepath, index=False)
        messagebox.showinfo("Exported", f"Forecast saved to:\n{filepath}")

    # ------------------------------------------------------------------ #
    # Tab 4: Evaluation
    # ------------------------------------------------------------------ #
    def _build_evaluation_tab(self):
        top = ttk.Frame(self.evaluation_tab, padding=10)
        top.pack(fill="x")
        self.mae_label = ttk.Label(top, text="MAE: —", font=("Segoe UI", 11, "bold"))
        self.mae_label.pack(side="left", padx=20)
        self.rmse_label = ttk.Label(top, text="RMSE: —", font=("Segoe UI", 11, "bold"))
        self.rmse_label.pack(side="left", padx=20)

        ttk.Label(
            self.evaluation_tab,
            text="Run '1. Train & Test Model' on the Forecast tab to populate this tab.",
            foreground="#777", padding=(10, 0),
        ).pack(anchor="w")

        ttk.Label(
            self.evaluation_tab,
            text="SARIMAX coefficient p-values (estimated on the training period; alpha = 0.05):",
            padding=(10, 6),
        ).pack(anchor="w")
        self.pvalues_tree = ttk.Treeview(
            self.evaluation_tab,
            columns=("term", "coefficient", "p_value", "interpretation"),
            show="headings",
            height=6,
        )
        for column, label, width in (
            ("term", "Model term", 180),
            ("coefficient", "Coefficient", 130),
            ("p_value", "P-value", 130),
            ("interpretation", "At alpha = 0.05", 220),
        ):
            self.pvalues_tree.heading(column, text=label)
            self.pvalues_tree.column(column, width=width, anchor="center")
        self.pvalues_tree.pack(fill="x", padx=10)
        ttk.Label(
            self.evaluation_tab,
            text="P-values describe evidence about model coefficients; they do not prove causation. "
                 "Use the final model specification and consult your adviser before reporting.",
            foreground="#777", wraplength=900, padding=10,
        ).pack(anchor="w")

        self.residual_chart_frame = ttk.Frame(self.evaluation_tab)
        self.residual_chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _update_evaluation(self, summary: dict, dates, actual, predicted):
        self.mae_label.config(text=f"MAE: {summary['MAE']}")
        self.rmse_label.config(text=f"RMSE: {summary['RMSE']}")
        self.pvalues_tree.delete(*self.pvalues_tree.get_children())
        for term, coefficient in self.fitted_model.params.items():
            if term not in self.fitted_model.pvalues.index:
                continue
            p_value = float(self.fitted_model.pvalues[term])
            if term in self._selected_exog_columns():
                interpretation = "Significant" if p_value < 0.05 else "Not significant"
            else:
                interpretation = "Model parameter"
            self.pvalues_tree.insert(
                "", "end",
                values=(term, f"{float(coefficient):.6g}", f"{p_value:.6g}", interpretation),
            )
        resid = evaluation.residuals(actual, predicted)
        fig = visualization.residuals_figure(dates, resid)
        visualization.embed_figure(fig, self.residual_chart_frame)


def run():
    root = tk.Tk()
    OasisApp(root)
    root.mainloop()
