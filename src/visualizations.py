from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import load_config

CONFIG = load_config()
THEME = CONFIG["theme"]

COLORWAY = [THEME["teal"], "#fff0cc", THEME["muted"], "#777a88", THEME["navy"]]


def apply_plotly_theme(fig, height: int = 420):
    fig.update_layout(
        height=height,
        colorway=COLORWAY,
        plot_bgcolor="#040406",
        paper_bgcolor="#040406",
        font={"color": THEME["text"], "family": THEME["font_family"]},
        margin={"l": 24, "r": 24, "t": 50, "b": 40},
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor=THEME["grid"], linecolor="#2e3038", zerolinecolor="#2e3038")
    fig.update_yaxes(gridcolor=THEME["grid"], linecolor="#2e3038", zerolinecolor="#2e3038")
    return fig


def actual_vs_predicted_chart(test_frame: pd.DataFrame, height: int = 420):
    fig = px.scatter(
        test_frame,
        x="actual",
        y="predicted",
        labels={"actual": "Actual kWh", "predicted": "Predicted kWh"},
        title="Actual vs Predicted Energy",
    )
    min_value = min(test_frame["actual"].min(), test_frame["predicted"].min())
    max_value = max(test_frame["actual"].max(), test_frame["predicted"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash", "color": THEME["navy"]},
        )
    )
    return apply_plotly_theme(fig, height)


def allocation_chart(loads: dict, allocation: dict, title: str, height: int = 420):
    frame = pd.DataFrame(
        [{"load": name.replace("_", " ").title(), "kWh": value} for name, value in allocation.items()]
    )
    fig = px.bar(frame, x="load", y="kWh", title=title, text_auto=".1f")
    return apply_plotly_theme(fig, height)


def comparison_chart(loads: dict, optimized: dict, baseline: dict, height: int = 460):
    rows = []
    for name in loads:
        label = name.replace("_", " ").title()
        rows.append({"load": label, "method": "Optimized", "kWh": optimized.get(name, 0.0)})
        rows.append({"load": label, "method": "Equal-cut baseline", "kWh": baseline.get(name, 0.0)})
    fig = px.bar(
        pd.DataFrame(rows),
        x="load",
        y="kWh",
        color="method",
        barmode="group",
        title="Baseline vs Optimized Allocation",
        text_auto=".1f",
    )
    return apply_plotly_theme(fig, height)


def feature_charts(df: pd.DataFrame, height: int = 340):
    fig = px.histogram(
        df,
        x="total_energy_kwh",
        nbins=int(THEME["histogram_bins"]),
        title="Synthetic Energy Generation Distribution",
        labels={"total_energy_kwh": "Generated kWh"},
    )
    return apply_plotly_theme(fig, height)
