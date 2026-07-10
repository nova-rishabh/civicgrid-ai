from __future__ import annotations

import base64
from copy import deepcopy
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.baseline import comparison_metrics, proportional_equal_cut
from src.config import get_loads, load_config
from src.data_generator import DATA_DIR, write_default_data
from src.export_utils import allocation_to_csv
from src.ml_model import predict_energy, train_model
from src.optimizer import optimize_allocation
from src.preprocessing import FEATURE_COLUMNS, load_csv, validate_features, validate_training_data
from src.visualizations import (
    actual_vs_predicted_chart,
    allocation_chart,
    comparison_chart,
    feature_charts,
)

ROOT = Path(__file__).resolve().parent
LOGO_PATH = Path(r"C:\Users\Admin\Downloads\Untitled (64 x 10 cm).png")
CONFIG = load_config()
THEME = CONFIG["theme"]
LOAD_DEMANDS = get_loads(CONFIG)
CRITICAL_LOAD = CONFIG["optimization"]["critical_load"]
CRITICAL_LABEL = LOAD_DEMANDS[CRITICAL_LOAD].get("label", CRITICAL_LOAD.replace("_", " ").title())
PAGES = CONFIG["app"]["pages"]

st.set_page_config(
    page_title=CONFIG["app"]["name"],
    page_icon=CONFIG["app"]["page_icon"],
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --color-obsidian:#08080a;
      --color-onyx:#040406;
      --color-carbon:#121317;
      --color-graphite:#1c1d22;
      --color-slate:#2e3038;
      --color-steel:#777a88;
      --color-fog:#9194a1;
      --color-mist:#acafb9;
      --color-bone:THEME_TEXT;
      --color-paper-white:#ffffff;
      --color-copper:THEME_COPPER;
      --color-error:THEME_ERROR;
      --gradient-gilded:linear-gradient(103deg, rgb(174,147,87), rgb(255,240,204) 40%, rgb(174,147,87) 70%, rgba(189,157,79,0));
      --font-serif:"Ivy Presto","Playfair Display","DM Serif Display",Georgia,serif;
      --font-ui:THEME_FONT;
      --radius-card:10px;
      --radius-pill:9999px;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
      background: var(--color-obsidian);
      color: var(--color-bone);
      font-family: var(--font-ui);
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stMainMenuButton"],
    [data-testid="stDeployButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarNav"],
    button[kind="header"],
    button[kind="headerNoPadding"],
    button[data-testid="stBaseButton-header"],
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="stMainMenuButton"] {
      display:none !important;
      visibility:hidden !important;
    }
    [data-testid="stMainBlockContainer"] {
      max-width: 1216px;
      padding-top: 100px;
      padding-bottom: 72px;
    }
    [data-testid="stSidebar"] {
      background: var(--color-onyx);
      border-right: 1px solid var(--color-graphite);
    }
    [data-testid="stSidebar"] * {
      color: var(--color-bone);
    }
    h1, h2, h3 {
      color: var(--color-paper-white);
      font-family: var(--font-serif);
      font-weight: 400;
      letter-spacing: 0;
    }
    h1 { font-size: clamp(3.1rem, 7vw, 5.5rem); line-height: 1; }
    h2 { font-size: clamp(2.2rem, 4.5vw, 4rem); line-height: 1.08; }
    h3 { font-size: clamp(1.65rem, 2.5vw, 2.75rem); line-height: 1.2; }
    p, li, label, [data-testid="stMarkdownContainer"] {
      color: var(--color-bone);
      font-family: var(--font-ui);
      letter-spacing: 0;
    }
    [data-testid="stCaptionContainer"], .muted, small {
      color: var(--color-fog) !important;
    }
    div[data-testid="stMetric"] {
      background: var(--color-onyx);
      border: 1px solid var(--color-graphite);
      border-radius: var(--radius-card);
      padding: 18px 20px;
      box-shadow: none;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
      color: var(--color-fog) !important;
    }
    div[data-testid="stMetricValue"] {
      color: var(--color-paper-white);
      font-family: var(--font-serif);
      letter-spacing: 0;
    }
    .stButton > button,
    .stDownloadButton > button {
      border-radius: var(--radius-pill);
      border: 1px solid var(--color-steel);
      background: transparent;
      color: var(--color-paper-white);
      padding: 10px 20px;
      font-weight: 500;
      transition: border-color .15s ease, background .15s ease, color .15s ease;
    }
    .stButton > button[kind="primary"] {
      background: var(--color-paper-white);
      color: #000000;
      border-color: var(--color-paper-white);
    }
    button[data-testid="stBaseButton-primary"] {
      background: var(--color-paper-white) !important;
      color: #000000 !important;
      border-color: var(--color-paper-white) !important;
    }
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-primary"] p {
      color: #000000 !important;
    }
    button[data-testid="stBaseButton-secondary"] {
      background: transparent !important;
      color: var(--color-paper-white) !important;
      border-color: var(--color-steel) !important;
    }
    button[data-testid="stBaseButton-secondary"] *,
    button[data-testid="stBaseButton-secondary"] p {
      color: var(--color-paper-white) !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
      border-color: var(--color-paper-white);
      color: var(--color-paper-white);
      background: var(--color-carbon);
    }
    .stButton > button[kind="primary"]:hover {
      background: var(--color-bone);
      color: #000000;
    }
    .badge {
      display:inline-block; padding:6px 10px; border-radius:8px; color:white;
      font-weight:700; background:transparent; border:1px solid var(--color-copper); color:var(--color-copper);
    }
    .badge.warn { border-color:var(--color-copper); color:var(--color-copper); }
    .badge.err { border-color:var(--color-error); color:var(--color-error); }
    .cg-topbar {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      z-index: 999999;
      border-bottom: 1px solid var(--color-graphite);
      background: rgba(4, 4, 6, 0.85);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
    }
    .cg-topbar-inner {
      max-width: 1152px;
      min-height: 68px;
      margin: 0 auto;
      padding: 0 24px;
      display: grid;
      grid-template-columns: minmax(170px, 1fr) auto minmax(190px, 1fr);
      align-items: center;
      gap: 28px;
    }
    .cg-brand {
      display:flex; align-items:center; min-width:0; color:var(--color-paper-white);
      text-decoration:none !important;
    }
    .cg-logo {
      display:block;
      width:154px;
      max-width:100%;
      height:auto;
      object-fit:contain;
    }
    .cg-nav {
      display:flex;
      align-items:center;
      justify-content:center;
      gap:28px;
      white-space:nowrap;
    }
    .cg-nav a,
    .cg-header-actions a {
      color:var(--color-paper-white) !important;
      text-decoration:none !important;
      font-size:14px;
      font-weight:600;
      line-height:1;
    }
    .cg-nav a {
      opacity:.88;
      transition:opacity .15s ease;
    }
    .cg-nav a:hover,
    .cg-nav a.active {
      opacity:1;
    }
    .cg-nav .chevron {
      color:var(--color-fog);
      font-size:12px;
      margin-left:4px;
    }
    .cg-header-actions {
      display:flex;
      align-items:center;
      justify-content:flex-end;
      gap:18px;
      white-space:nowrap;
    }
    .cg-header-cta {
      border:1px solid var(--color-steel);
      border-radius:9999px;
      padding:10px 16px;
      transition:border-color .15s ease, background .15s ease;
    }
    .cg-header-cta:hover {
      border-color:var(--color-paper-white);
      background:var(--color-carbon);
    }
    .cg-page-buttons [data-testid="column"] {
      width:auto !important;
      flex:0 1 auto !important;
    }
    .cg-page-buttons button {
      min-height:34px;
      padding:6px 12px !important;
      font-size:14px;
    }
    .cg-helper {
      margin:8px 0 24px;
      color:var(--color-fog);
      font-size:14px;
      line-height:1.5;
    }
    .cg-info-panel {
      border:1px solid var(--color-graphite);
      background:var(--color-onyx);
      border-radius:var(--radius-card);
      padding:18px 20px;
      margin:16px 0 24px;
    }
    .cg-info-panel strong {
      display:block;
      color:var(--color-paper-white);
      font-size:16px;
      margin-bottom:6px;
    }
    .cg-info-panel p {
      color:var(--color-fog);
      font-size:14px;
      line-height:1.55;
      margin:0;
    }
    .cg-control-strip {
      border:1px solid var(--color-graphite);
      background:var(--color-onyx);
      border-radius:var(--radius-card);
      padding:18px 20px;
      margin:14px 0 24px;
    }
    .cg-control-strip h3 {
      font-family:var(--font-ui);
      font-size:16px;
      font-weight:600;
      margin:0 0 6px;
      color:var(--color-paper-white);
    }
    .cg-control-strip p {
      color:var(--color-fog);
      font-size:14px;
      line-height:1.5;
      margin:0;
    }
    .cg-explain-grid {
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:16px;
      margin-top:26px;
    }
    .cg-judge-card {
      background:var(--color-onyx);
      border:1px solid var(--color-graphite);
      border-radius:var(--radius-card);
      padding:22px;
    }
    .cg-judge-card h3 {
      font-family:var(--font-ui);
      font-size:18px;
      font-weight:600;
      margin:0 0 10px;
      color:var(--color-paper-white);
    }
    .cg-judge-card p,
    .cg-judge-card li {
      color:var(--color-fog);
      font-size:15px;
      line-height:1.55;
    }
    .cg-judge-card ul {
      margin:0;
      padding-left:18px;
    }
    .cg-eyebrow {
      color:var(--color-copper); font-size:13px; font-weight:600; text-transform:uppercase;
      letter-spacing:0; margin-bottom:16px;
    }
    .cg-hero {
      display:grid; grid-template-columns:minmax(0, 1.05fr) minmax(360px, 0.95fr);
      gap:48px; align-items:center; min-height:620px; padding:52px 0 76px;
      background-image: 
        radial-gradient(circle at 85% 30%, rgba(212, 175, 98, 0.06) 0%, transparent 55%),
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
      background-size: 100% 100%, 32px 32px, 32px 32px;
    }
    .cg-hero h1 {
      margin:0; color:var(--color-paper-white);
      font-family:var(--font-serif); font-size:clamp(38px, 4.5vw, 56px);
      line-height:1.05; font-weight:400; letter-spacing:-0.01em;
    }
    .cg-hero-copy {
      max-width:650px; color:var(--color-fog); font-size:20px; line-height:1.5;
      margin:24px 0 0;
    }
    .cg-actions { display:flex; gap:12px; flex-wrap:wrap; margin-top:30px; align-items:center; }
    .cg-primary, .cg-secondary {
      display:inline-flex; align-items:center; justify-content:center; border-radius:9999px;
      padding:11px 20px; font-size:14px; font-weight:500; text-decoration:none !important;
    }
    .cg-primary { background:var(--color-paper-white); color:#000 !important; }
    .cg-secondary { color:var(--color-paper-white) !important; border:1px solid var(--color-steel); }
    .cg-console {
      background:var(--color-onyx); border:1px solid var(--color-graphite);
      border-radius:var(--radius-card); padding:22px; min-height:420px;
    }
    .cg-console-head { display:flex; justify-content:space-between; gap:16px; align-items:start; margin-bottom:22px; }
    .cg-console-label { color:var(--color-fog); font-size:13px; }
    .cg-console-value { font-size:48px; line-height:1; color:var(--color-paper-white); font-weight:500; margin-top:8px; }
    .cg-pill {
      border:1px solid var(--color-steel); border-radius:9999px; padding:6px 10px;
      color:var(--color-bone); font-size:12px; white-space:nowrap;
    }
    .cg-chart {
      height:146px; border-bottom:1px solid var(--color-graphite); position:relative;
      margin:20px 0 18px; overflow:hidden;
    }
    .cg-chart::before {
      content:""; position:absolute; inset:0;
      background:
        linear-gradient(to right, transparent 0, transparent 24%, var(--color-graphite) 24.2%, transparent 24.5%),
        linear-gradient(to bottom, transparent 0, transparent 32%, rgba(28,29,34,.72) 32.5%, transparent 33%),
        linear-gradient(to bottom, transparent 0, transparent 66%, rgba(28,29,34,.72) 66.5%, transparent 67%);
      opacity:.8;
    }
    .cg-chart-line {
      position:absolute; left:0; right:0; top:40px; height:72px;
      background:var(--gradient-gilded);
      clip-path:polygon(0 72%, 12% 64%, 23% 68%, 34% 42%, 48% 48%, 62% 24%, 75% 36%, 88% 12%, 100% 20%, 100% 27%, 88% 19%, 75% 43%, 62% 31%, 48% 55%, 34% 49%, 23% 75%, 12% 71%, 0 79%);
    }
    .cg-row {
      display:grid; grid-template-columns:1fr auto; gap:16px; align-items:center;
      border-bottom:1px solid var(--color-graphite); padding:12px 0;
    }
    .cg-row:last-child { border-bottom:0; }
    .cg-row-title { color:var(--color-bone); font-size:15px; }
    .cg-row-meta { color:var(--color-fog); font-size:13px; margin-top:3px; }
    .cg-row-value { color:var(--color-paper-white); font-size:15px; text-align:right; }
    .cg-section { padding:72px 0 20px; }
    .cg-section-title {
      font-family:var(--font-serif); color:var(--color-paper-white);
      font-size:clamp(28px, 3.5vw, 42px); line-height:1.15; letter-spacing:-0.005em; margin:0 0 16px;
      max-width:820px;
    }
    .cg-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; margin-top:28px; }
    .cg-card {
      background:var(--color-onyx); border:1px solid var(--color-graphite);
      border-radius:var(--radius-card); padding:24px;
    }
    .cg-card h3 {
      font-family:var(--font-ui); font-size:20px; font-weight:500; line-height:1.35;
      color:var(--color-paper-white); margin:0 0 10px;
    }
    .cg-card p { color:var(--color-fog); font-size:15px; line-height:1.55; margin:0; }
    .cg-stat-number {
      font-family:var(--font-serif); font-size:44px; color:var(--color-paper-white);
      line-height:1; letter-spacing:0;
    }
    .cg-stat-caption { color:var(--color-fog); font-size:14px; margin-top:10px; }
    .cg-flow {
      display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; margin-top:28px;
    }
    .cg-step { border-top:1px solid var(--color-graphite); padding-top:18px; }
    .cg-step-index { color:var(--color-copper); font-size:13px; font-weight:600; margin-bottom:10px; }
    .cg-step-title { color:var(--color-paper-white); font-size:20px; font-weight:500; margin-bottom:8px; }
    .cg-step-copy { color:var(--color-fog); font-size:15px; line-height:1.5; }
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
      border:1px solid var(--color-graphite); border-radius:var(--radius-card); overflow:hidden;
    }
    .stAlert {
      background:var(--color-carbon);
      border:1px solid var(--color-graphite);
      border-radius:var(--radius-card);
      color:var(--color-bone);
    }
    iframe { border-radius:var(--radius-card); }
    @media (max-width: 900px) {
      [data-testid="stMainBlockContainer"] {
        padding-top: 24px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
      }
      .cg-hero { grid-template-columns:1fr; min-height:auto; padding-top:30px; }
      .cg-grid, .cg-flow, .cg-explain-grid { grid-template-columns:1fr; }
      .cg-console { min-height:auto; }
      .cg-topbar {
        position: static;
        width: 100%;
        margin-top: 0;
        margin-bottom: 28px;
        background: var(--color-onyx);
        backdrop-filter: none;
      }
      .cg-topbar-inner {
        grid-template-columns:1fr;
        justify-items:start;
        gap:18px;
        padding:18px;
      }
      .cg-nav {
        flex-wrap:wrap;
        justify-content:flex-start;
        gap:14px 20px;
      }
      .cg-header-actions {
        justify-content:flex-start;
      }
    }
    </style>
    """
    .replace("THEME_TEXT", THEME["text"])
    .replace("THEME_COPPER", THEME["teal"])
    .replace("THEME_ERROR", THEME["error"])
    .replace("THEME_FONT", THEME["font_family"]),
    unsafe_allow_html=True,
)


def ensure_data() -> None:
    write_default_data(CONFIG)


def load_training(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_csv(DATA_DIR / "training_data.csv")


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def page_url(page_name: str) -> str:
    pres = "true" if st.session_state.get("presentation_mode", False) else "false"
    scen = st.session_state.get("selected_scenario", CONFIG["app"]["default_scenario"])
    return f"?page={quote(page_name)}&presentation_mode={pres}&scenario={quote(scen)}"


def scenario_features(name: str, feature_override: dict | None = None) -> tuple[dict, float | None, float]:
    scenario_config = CONFIG["scenarios"][name]
    row = load_csv(DATA_DIR / scenario_config["filename"]).iloc[0].to_dict()
    forced = row.get("forced_predicted_energy")
    multiplier = float(row.get("demand_multiplier", 1.0))
    features = feature_override or {column: row[column] for column in FEATURE_COLUMNS}
    return features, None if pd.isna(forced) else float(forced), multiplier


def scaled_loads(multiplier: float) -> dict:
    loads = deepcopy(LOAD_DEMANDS)
    if multiplier == 1:
        return loads
    for name, load in loads.items():
        load["demand"] = round(float(load["demand"]) * multiplier, 2)
        if CONFIG["data_generation"]["scenario_scale_min_required"]:
            load["min_required"] = round(float(load["min_required"]) * multiplier, 2)
    return loads


def run_workflow(
    training_df: pd.DataFrame,
    scenario: str,
    edited_loads: dict | None = None,
    feature_override: dict | None = None,
):
    result = train_model(training_df, None)
    features, forced_energy, multiplier = scenario_features(scenario, feature_override)
    predicted = forced_energy if forced_energy is not None else predict_energy(features, result["model"])
    loads = edited_loads or scaled_loads(multiplier)
    optimized = optimize_allocation(predicted, loads)
    baseline = proportional_equal_cut(predicted, loads)
    metrics = comparison_metrics(optimized["allocation"], baseline, loads, CRITICAL_LOAD)
    return result, features, predicted, loads, optimized, baseline, metrics


def load_label(loads: dict, name: str) -> str:
    return loads[name].get("label", name.replace("_", " ").title())


def render_topbar(active_page: str) -> None:
    logo_uri = image_data_uri(LOGO_PATH)
    brand_html = (
        f'<img class="cg-logo" src="{logo_uri}" alt="CivicGrid AI">'
        if logo_uri
        else f'<span>{escape(CONFIG["app"]["name"])}</span>'
    )
    nav_items = [
        ("Data & Prediction", PAGES[1], False),
        ("Optimization", PAGES[2], False),
        ("Results & Impact", PAGES[3], False),
        ("Mathematics & About", PAGES[4], False),
    ]
    nav_parts = []
    for label, target, has_chevron in nav_items:
        active_class = "active" if active_page == target else ""
        chevron = '<span class="chevron">&#8964;</span>' if has_chevron else ""
        nav_parts.append(f'<a class="{active_class}" href="{page_url(target)}">{escape(label)}{chevron}</a>')
    nav_html = "".join(nav_parts)
    st.markdown(
        f"""
        <div class="cg-topbar">
          <div class="cg-topbar-inner">
            <a class="cg-brand" href="{page_url(PAGES[0])}" aria-label="CivicGrid AI Home">{brand_html}</a>
            <nav class="cg-nav" aria-label="Primary navigation">{nav_html}</nav>
            <div class="cg-header-actions">
              <a class="cg-header-cta" href="{page_url(PAGES[1])}">Get Started</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_console_svg(allocation: dict, baseline: dict, loads: dict) -> str:
    ordered_loads = sorted(loads.keys(), key=lambda name: loads[name]["priority"], reverse=True)
    max_demand = max(1.0, max(float(load["demand"]) for load in loads.values()))
    max_alloc = max(0.1, max(float(val) for val in allocation.values()))
    max_y_val = max(max_demand, max_alloc)
    
    x_coords = [20, 100, 180, 260, 340]
    y_coords = []
    y_coords_base = []
    for name in ordered_loads:
        val = float(allocation.get(name, 0.0))
        y = 95 - (val / max_y_val) * 80
        y_coords.append(round(y, 1))
        
        val_base = float(baseline.get(name, 0.0))
        y_base = 95 - (val_base / max_y_val) * 80
        y_coords_base.append(round(y_base, 1))
        
    path_data = f"M {x_coords[0]},{y_coords[0]} " + " ".join(f"L {x},{y}" for x, y in zip(x_coords[1:], y_coords[1:]))
    area_data = f"M {x_coords[0]},95 L {x_coords[0]},{y_coords[0]} " + " ".join(f"L {x},{y}" for x, y in zip(x_coords[1:], y_coords[1:])) + f" L {x_coords[-1]},95 Z"
    
    path_data_base = f"M {x_coords[0]},{y_coords_base[0]} " + " ".join(f"L {x},{y}" for x, y in zip(x_coords[1:], y_coords_base[1:]))
    
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="3" fill="#ffffff" stroke="rgb(174,147,87)" stroke-width="1.5" />' for x, y in zip(x_coords, y_coords))
    dots_base = "".join(f'<circle cx="{x}" cy="{y}" r="2" fill="#ffffff" stroke="rgba(100, 149, 237, 0.7)" stroke-width="1" />' for x, y in zip(x_coords, y_coords_base))
    
    grid_lines = """
    <line x1="20" y1="15" x2="340" y2="15" stroke="#1c1d22" stroke-dasharray="2,2" />
    <line x1="20" y1="55" x2="340" y2="55" stroke="#1c1d22" stroke-dasharray="2,2" />
    <line x1="20" y1="95" x2="340" y2="95" stroke="#1c1d22" stroke-dasharray="2,2" />
    """
    
    legend = """
    <g transform="translate(250, 10)" style="font-size: 8px; font-family: var(--font-ui), sans-serif; font-weight: 500;">
      <line x1="0" y1="3" x2="10" y2="3" stroke="rgb(174,147,87)" stroke-width="2" />
      <text x="14" y="6" fill="#ffffff">Optimized</text>
      <line x1="0" y1="13" x2="10" y2="13" stroke="rgba(100, 149, 237, 0.7)" stroke-width="1.5" stroke-dasharray="2,2" />
      <text x="14" y="16" fill="#8e909a">Equal-Cut</text>
    </g>
    """
    
    svg = f"""
    <svg viewBox="0 0 360 110" width="100%" height="110" xmlns="http://www.w3.org/2000/svg" style="display:block; overflow:visible;">
      <defs>
        <linearGradient id="svg-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(174,147,87,0.45)" />
          <stop offset="100%" stop-color="rgba(174,147,87,0.0)" />
        </linearGradient>
      </defs>
      {grid_lines}
      <path d="{area_data}" fill="url(#svg-gradient)" />
      <path d="{path_data_base}" fill="none" stroke="rgba(100, 149, 237, 0.7)" stroke-width="1.5" stroke-dasharray="3,3" />
      <path d="{path_data}" fill="none" stroke="rgb(174,147,87)" stroke-width="2" />
      {dots_base}
      {dots}
      {legend}
    </svg>
    """
    return svg.replace("\n", " ").strip()


def render_landing_hero(
    predicted_energy: float,
    loads: dict,
    optimized: dict,
    baseline: dict,
    metrics: dict,
    scenario_name: str,
) -> None:
    show_results = st.session_state.get("show_results", True)
    if show_results:
        allocation = optimized["allocation"]
        top_rows = sorted(loads, key=lambda name: loads[name]["priority"], reverse=True)[:4]
        row_html = "".join(
            f'<div class="cg-row"><div><div class="cg-row-title">{escape(load_label(loads, name))}</div><div class="cg-row-meta">priority {loads[name]["priority"]:.0f} / demand {loads[name]["demand"]:.1f} kWh</div></div><div class="cg-row-value" style="display:flex; flex-direction:column; align-items:flex-end;"><span style="color:rgb(212,175,98); font-weight:600;">{allocation.get(name, 0):.1f} kWh</span><span style="color:var(--color-fog); font-size:12px;">{baseline.get(name, 0):.1f} kWh baseline</span></div></div>'
            for name in top_rows
        )
        right_panel = f"""
          <div class="cg-console" aria-label="CivicGrid allocation preview">
            <div class="cg-console-head">
              <div>
                <div class="cg-console-label">Predicted available energy</div>
                <div class="cg-console-value">{predicted_energy:.1f} kWh</div>
              </div>
              <div class="cg-pill">{escape(optimized["status"])}</div>
            </div>
            <div style="height:110px; margin:20px 0 18px; border-bottom:1px solid var(--color-graphite);">{generate_console_svg(allocation, baseline, loads)}</div>
            {row_html}
          </div>
        """
    else:
        bg_img_path = Path(r"C:\Users\Admin\.gemini\antigravity\brain\d999e81f-eb90-4d6a-b12f-34a58f3f8c80\slash_hero_bg_1783614341883.png")
        bg_uri = image_data_uri(bg_img_path) or ""
        right_panel = f"""
          <div class="cg-console" style="border:none; background:none; padding:0; display:flex; justify-content:center; align-items:center; min-height:360px; box-sizing:border-box;">
            <img src="{bg_uri}" style="width:100%; border-radius:12px; border:1px solid var(--color-graphite); box-shadow: 0 20px 50px rgba(0,0,0,0.6);" alt="Neural Microgrid Operating System" />
          </div>
        """

    st.markdown(
        f"""
        <section class="cg-hero">
          <div>
            <div class="cg-eyebrow">Microgrid allocation intelligence</div>
            <h1>Predict power. Protect what matters.</h1>
            <p class="cg-hero-copy">
              {escape(CONFIG["app"]["problem_statement"])}
            </p>
            <div class="cg-actions">
              <a class="cg-primary" href="{page_url(PAGES[1])}">Start the judge demo</a>
              <a class="cg-secondary" href="{page_url(PAGES[3])}">View impact</a>
              <span class="cg-pill">Scenario: {escape(scenario_name)}</span>
            </div>
          </div>
          {right_panel}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_landing_details(
    predicted_energy: float,
    loads: dict,
    optimized: dict,
    baseline: dict,
    metrics: dict,
) -> None:
    allocation = optimized["allocation"]
    total_demand = sum(float(load["demand"]) for load in loads.values())
    shortfall = max(total_demand - predicted_energy, 0)
    critical_metric = next(label for label in metrics if label.endswith("minimum met (%)"))
    baseline_critical = metrics[critical_metric]["baseline"]
    optimized_critical = metrics[critical_metric]["optimized"]
    st.markdown(
        f"""
        <section class="cg-section">
          <div class="cg-eyebrow">What judges should notice</div>
          <div class="cg-section-title">This is not just a dashboard. It is an ML forecast wired into an optimization decision.</div>
          <div class="cg-explain-grid">
            <div class="cg-judge-card">
              <h3>The civic problem</h3>
              <p>During a shortfall, a naive equal split can reduce power to safety-critical services. CivicGrid AI makes the shortage visible and then allocates scarce energy by priority.</p>
            </div>
            <div class="cg-judge-card">
              <h3>The technical idea</h3>
              <p>The Linear Regression model predicts next-hour <strong style="display:inline;color:var(--color-paper-white);">total_energy_kwh</strong>. That exact number becomes the supply limit in the PuLP linear program.</p>
            </div>
            <div class="cg-judge-card">
              <h3>The demo path</h3>
              <ul>
                <li>Open Data & Prediction to train the model.</li>
                <li>Open Optimization to edit load demands and run the solver.</li>
                <li>Open Results & Impact to compare against equal-cut allocation.</li>
              </ul>
            </div>
            <div class="cg-judge-card">
              <h3>Why it matters</h3>
              <p>The optimized result keeps {escape(CRITICAL_LABEL)} protected whenever the forecasted supply can satisfy its minimum, while lower-priority loads share the remaining energy.</p>
            </div>
          </div>
        </section>
        <section class="cg-section">
          <div class="cg-eyebrow">Operational impact</div>
          <div class="cg-section-title">A civic grid demo that shows the decision, not just the data.</div>
          <div class="cg-grid">
            <div class="cg-card">
              <div class="cg-stat-number">{optimized_critical:.0f}%</div>
              <div class="cg-stat-caption">{escape(CRITICAL_LABEL)} minimum met by the optimizer</div>
            </div>
            <div class="cg-card">
              <div class="cg-stat-number">{baseline_critical:.1f}%</div>
              <div class="cg-stat-caption">minimum met by equal-cut baseline</div>
            </div>
            <div class="cg-card">
              <div class="cg-stat-number">{shortfall:.1f}</div>
              <div class="cg-stat-caption">kWh shortfall against total demand of {total_demand:.1f} kWh</div>
            </div>
          </div>
        </section>
        <section class="cg-section">
          <div class="cg-eyebrow">Predict -> Optimize -> Impact</div>
          <div class="cg-flow">
            <div class="cg-step">
              <div class="cg-step-index">01</div>
              <div class="cg-step-title">Forecast generation</div>
              <div class="cg-step-copy">Linear regression estimates next-hour energy from solar, footfall, temperature, hour, and weekend context.</div>
            </div>
            <div class="cg-step">
              <div class="cg-step-index">02</div>
              <div class="cg-step-title">Allocate by priority</div>
              <div class="cg-step-copy">The predicted value becomes the LP supply constraint, while critical minimums remain protected.</div>
            </div>
            <div class="cg-step">
              <div class="cg-step-index">03</div>
              <div class="cg-step-title">Compare outcomes</div>
              <div class="cg-step-copy">The app contrasts optimized allocation with equal-cut baseline using clear operational metrics.</div>
            </div>
          </div>
        </section>
        <section class="cg-section">
          <div class="cg-eyebrow">Team</div>
          <div class="cg-card"><p>{escape(CONFIG["app"]["team"])}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


ensure_data()

if "demo_ran" not in st.session_state:
    st.session_state.demo_ran = False
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "show_results" not in st.session_state:
    st.session_state.show_results = True
requested_page = st.query_params.get("page")
if requested_page in PAGES:
    st.session_state.page = requested_page
else:
    st.session_state.page = PAGES[0]
    st.query_params["page"] = PAGES[0]

scenario_names = list(CONFIG["scenarios"].keys())
default_scenario = CONFIG["app"]["default_scenario"]
default_index = scenario_names.index(default_scenario) if default_scenario in scenario_names else 0

query_scenario = st.query_params.get("scenario")
if query_scenario in scenario_names:
    st.session_state.selected_scenario = query_scenario
scenario = st.session_state.get("selected_scenario", default_scenario)

if st.query_params.get("scenario") != scenario:
    st.query_params["scenario"] = scenario
uploaded = None
base_scenario_row = load_csv(DATA_DIR / CONFIG["scenarios"][scenario]["filename"]).iloc[0]
custom_features = {column: base_scenario_row[column] for column in FEATURE_COLUMNS}

page = st.session_state.page
render_topbar(page)
page = st.session_state.page
hero_container = st.container() if page == PAGES[0] else None

query_pres = st.query_params.get("presentation_mode") == "true"
if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = query_pres

col_helper, col_toggle = st.columns([0.8, 0.2])
with col_toggle:
    presentation_mode = st.toggle("Presentation Mode", value=st.session_state.presentation_mode, key="presentation_mode")

if st.query_params.get("presentation_mode") != ("true" if presentation_mode else "false"):
    st.query_params["presentation_mode"] = "true" if presentation_mode else "false"

if page != PAGES[0]:
    with col_helper:
        st.markdown(
            '<div class="cg-helper" style="margin-top: 8px;">Use the page selector above to move through the judging demo. Each page explains the step it performs.</div>',
            unsafe_allow_html=True,
        )

if not presentation_mode:
    st.markdown(
        """
        <div class="cg-control-strip">
          <h3>🏛️ Judge Sandbox Simulator</h3>
          <p>Tweak next-hour weather and footfall features in real-time. Use the Reference Scenarios on the right as a guide. The forecast model is preserved, and the system instantly re-predicts and re-allocates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "solar_val" not in st.session_state:
        row = load_csv(DATA_DIR / CONFIG["scenarios"][scenario]["filename"]).iloc[0]
        st.session_state.solar_val = float(row["solar_irradiance"])
        st.session_state.footfall_val = int(row["footfall_count"])
        st.session_state.temp_val = float(row["temperature"])
        st.session_state.hour_val = int(row["hour_of_day"])
        st.session_state.weekend_val = bool(row["is_weekend"])

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Simulator Inputs")
        current_index = scenario_names.index(scenario) if scenario in scenario_names else default_index
        def handle_scenario_select():
            scen = st.session_state.selected_scenario_select
            st.session_state.selected_scenario = scen
            row = load_csv(DATA_DIR / CONFIG["scenarios"][scen]["filename"]).iloc[0]
            st.session_state.solar_val = float(row["solar_irradiance"])
            st.session_state.footfall_val = int(row["footfall_count"])
            st.session_state.temp_val = float(row["temperature"])
            st.session_state.hour_val = int(row["hour_of_day"])
            st.session_state.weekend_val = bool(row["is_weekend"])

        scenario = st.selectbox(
            "Load Preset Scenario",
            scenario_names,
            index=current_index,
            key="selected_scenario_select",
            on_change=handle_scenario_select
        )
        
        st.slider("Solar Irradiance (W/m²)", 0, 1000, key="solar_val")
        st.slider("Footfall Count (people/hour)", 0, 500, key="footfall_val")
        st.slider("Temperature (°C)", 15, 45, key="temp_val")
        st.number_input("Hour of Day (24h)", 0, 23, key="hour_val")
        st.toggle("Weekend Hour?", key="weekend_val")

        custom_features = {
            "hour_of_day": int(st.session_state.hour_val),
            "solar_irradiance": float(st.session_state.solar_val),
            "footfall_count": int(st.session_state.footfall_val),
            "temperature": float(st.session_state.temp_val),
            "is_weekend": 1 if st.session_state.weekend_val else 0,
        }
        
        with st.expander("Upload custom training CSV (Optional)"):
            uploaded = st.file_uploader("Upload training CSV", type="csv")
            st.caption("Uploaded CSVs must match the displayed schema. Leave blank to use the bundled improved training dataset.")

    with right_col:
        st.subheader("Reference Demo Data")
        st.markdown(
            """
            <div class="cg-info-panel" style="margin:0; height:100%; box-sizing:border-box;">
              <strong>🎯 Scenario Guide for Judges:</strong>
              <table style="width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; color:var(--color-fog);">
                <thead>
                  <tr style="border-bottom:1px solid var(--color-graphite); text-align:left;">
                    <th style="padding:6px 0; font-weight:600;">Scenario</th>
                    <th style="padding:6px 0; font-weight:600;">Inputs to Enter</th>
                    <th style="padding:6px 0; font-weight:600;">Expected Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:8px 0; color:var(--color-paper-white); font-weight:600;">Demo</td>
                    <td style="padding:8px 0;">Solar: 650<br>Footfall: 220<br>Temp: 31<br>Weekend: No</td>
                    <td style="padding:8px 0; color:rgb(212,175,98);">Optimal (42.5 kWh)<br>All critical loads met.</td>
                  </tr>
                  <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:8px 0; color:var(--color-paper-white); font-weight:600;">High Demand</td>
                    <td style="padding:8px 0;">Solar: 900<br>Footfall: 450<br>Temp: 34<br>Weekend: No</td>
                    <td style="padding:8px 0; color:rgb(212,175,98);">Optimal (62.1 kWh)<br>Charging throttled, displays cut.</td>
                  </tr>
                  <tr>
                    <td style="padding:8px 0; color:var(--color-paper-white); font-weight:600;">Infeasible</td>
                    <td style="padding:8px 0;">Solar: 0<br>Footfall: 10<br>Temp: 20<br>Weekend: Yes</td>
                    <td style="padding:8px 0; color:#ff4b4b; font-weight:600;">Infeasible (3.0 kWh)<br>Safety limits violated.</td>
                  </tr>
                </tbody>
              </table>
              <p style="margin-top:14px; font-size:12px; line-height:1.45; color:var(--color-fog);">
                💡 <b>Try it:</b> Drag the sliders to match any of the scenarios, click <b>Run Complete Demo</b>, and watch the top console chart and bottom impact stats update immediately!
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

training_df = load_training(uploaded)
validation_errors = validate_training_data(training_df)
feature_errors = validate_features(pd.DataFrame([custom_features]))

if validation_errors:
    st.error("Training data needs attention: " + " ".join(validation_errors))
    st.stop()
if feature_errors:
    st.error("Scenario inputs need attention: " + " ".join(feature_errors))
    st.stop()

input_signature = (
    scenario,
    tuple((column, float(custom_features[column])) for column in FEATURE_COLUMNS),
    uploaded.name if uploaded is not None else "bundled",
)

if st.session_state.get("active_input_signature") != input_signature:
    st.session_state.workflow = None
    st.session_state.active_input_signature = input_signature
    st.session_state.last_action = f"Inputs changed. Workflow refreshed with scenario {scenario}."
    st.session_state.show_results = False

if st.session_state.workflow is None:
    st.session_state.workflow = run_workflow(training_df, scenario, feature_override=custom_features)

model_result, features, predicted_energy, loads, optimized, baseline, metrics = st.session_state.workflow

if not presentation_mode:
    control_col,reset_col, _ = st.columns([0.2, 0.14, 0.66])
    with control_col:
        run_demo = st.button("Run Complete Demo", type="primary", use_container_width=True)
    with reset_col:
        reset_demo = st.button("Reset", use_container_width=True)

    if run_demo:
        import time
        with st.status("Executing CivicGrid Pipeline...", expanded=True) as status:
            st.write("📂 Loading microgrid training dataset...")
            time.sleep(0.4)
            st.write("🧠 Training Linear Regression model on historical weather & footfall...")
            time.sleep(0.4)
            st.write("🔮 Predicting next-hour available energy...")
            time.sleep(0.4)
            st.write("⚡ Formulating and solving PuLP Linear Program...")
            time.sleep(0.4)
            st.session_state.workflow = run_workflow(training_df, scenario, feature_override=custom_features)
            st.session_state.demo_ran = True
            st.session_state.show_results = True
            st.session_state.last_action = "Complete demo ran: data loaded, model trained, energy predicted, LP optimized, and impact metrics refreshed."
            status.update(label="CivicGrid Pipeline executed successfully!", state="complete", expanded=False)
        st.rerun()

    if reset_demo:
        st.session_state.workflow = None
        st.session_state.demo_ran = False
        st.session_state.show_results = False
        st.session_state.selected_scenario = default_scenario
        st.session_state.last_action = "Reset complete: simulator inputs returned to defaults."
        
        # Safely delete widget-tied keys so Streamlit recreates them with defaults on rerun
        for key in ["solar_val", "footfall_val", "temp_val", "hour_val", "weekend_val", "selected_scenario_select"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown(
        """
        <div class="cg-info-panel">
          <strong>What do these buttons do?</strong>
          <p><b>Run Complete Demo</b> performs the full pipeline in one click: train model -> predict energy -> optimize allocation -> compare impact. <b>Reset</b> clears the current run and reloads the default demo scenario.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_action:
        st.info(st.session_state.last_action)

    if st.session_state.get("show_results", True):
        # Build allocation rows HTML for Live Simulation Output
        allocation_html_rows = ""
        for name in loads.keys():
            label = loads[name].get("label", name.replace("_", " ").title())
            demanded = loads[name]["demand"]
            allocated = optimized["allocation"].get(name, 0.0)
            percentage = min(100.0, round(allocated / demanded * 100, 1)) if demanded > 0 else 0.0
            
            # Determine status text and color
            if percentage >= 99.9:
                status_text = "Protected ✅"
                status_color = "#4caf50"
            elif percentage > 0.0:
                status_text = f"Throttled ({percentage:.0f}%) ⚠️"
                status_color = "#ff9800"
            else:
                status_text = "Shed ❌"
                status_color = "#f44336"
                
            allocation_html_rows += (
                f'<div style="display:flex; flex-direction:column; gap:4px; margin-bottom:12px;">'
                f'  <div style="display:flex; justify-content:space-between; font-size:13px; line-height:1.2;">'
                f'    <strong style="color:var(--color-paper-white);">{label}</strong>'
                f'    <span style="color:{status_color}; font-weight:600;">{allocated:.1f} / {demanded:.1f} kWh ({status_text})</span>'
                f'  </div>'
                f'  <div style="background:var(--color-graphite); height:6px; border-radius:3px; overflow:hidden; width:100%;">'
                f'    <div style="background:rgb(212,175,98); height:6px; width:{percentage}%; border-radius:3px;"></div>'
                f'  </div>'
                f'</div>'
            )

        status_pill_class = "badge" if optimized["status"] == "Optimal" else "badge err"
        st.markdown(
            f'<div class="cg-control-strip" style="margin-top:20px; border-color:var(--color-graphite);">'
            f'  <h3 style="margin-bottom:16px;">📊 Live Simulation Output</h3>'
            f'  <div style="display:grid; grid-template-columns: 1fr 1.5fr; gap:20px;">'
            f'    <!-- Left: Model Predictions -->'
            f'    <div class="cg-info-panel" style="margin:0; height:100%; box-sizing:border-box;">'
            f'      <strong style="font-size:14px; color:var(--color-paper-white);">ML Next-Hour Forecast</strong>'
            f'      <div style="font-size:32px; font-weight:700; color:rgb(212,175,98); margin:12px 0 6px 0; font-family:var(--font-heading);">{predicted_energy:.2f} kWh</div>'
            f'      <p style="font-size:12px; line-height:1.4; margin-bottom:16px; color:var(--color-fog);">'
            f'        This is the predicted energy limit generated from your inputs.'
            f'      </p>'
            f'      <div style="display:flex; align-items:center; gap:8px;">'
            f'        <span style="font-size:12px; color:var(--color-fog);">Solver Status:</span>'
            f'        <span class="{status_pill_class}" style="font-size:11px; padding:3px 8px; border-radius:10px;">{optimized["status"]}</span>'
            f'      </div>'
            f'    </div>'
            f'    <!-- Right: Load Allocations -->'
            f'    <div class="cg-info-panel" style="margin:0; height:100%; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center;">'
            f'      <strong style="font-size:14px; color:var(--color-paper-white); margin-bottom:14px; display:block;">Optimized Load Allocation</strong>'
            f'      {allocation_html_rows}'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:rgba(212,175,98,0.04); border:1px solid rgba(212,175,98,0.15); padding:18px; border-radius:8px; margin-top:20px; text-align:center;">
              <span style="color:rgb(212,175,98); font-weight:600; font-size:14px;">⚠️ Inputs changed. Click "Run Complete Demo" above to solve and display the allocation results.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
chart_height = int(THEME["presentation_chart_height"]) if presentation_mode else int(THEME["chart_height"])
critical_metric_label = next(label for label in metrics if label.endswith("minimum met (%)"))

if page == PAGES[0]:
    with hero_container:
        render_landing_hero(predicted_energy, loads, optimized, baseline, metrics, scenario)
    render_landing_details(predicted_energy, loads, optimized, baseline, metrics)

elif page == PAGES[1]:
    st.markdown('<div class="cg-eyebrow">Data & Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="cg-section-title">Train the forecast model and inspect prediction quality.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="cg-info-panel">
          <strong>What does Train Model do?</strong>
          <p>It trains a Linear Regression model on the displayed training data, tests it on a held-out split, and refreshes the prediction used later by the optimizer. In judge mode this is kept in memory so the packaged model file is not overwritten by experiments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(training_df.head(int(THEME["preview_rows"])), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Validation")
        st.success("Training CSV matches the expected schema.")
        st.plotly_chart(feature_charts(training_df, chart_height - 80), use_container_width=True)

    if st.button("Train Model", use_container_width=True):
        import time
        with st.status("Training Forecast Model...", expanded=True) as status:
            st.write("📂 Preparing features and splitting dataset...")
            time.sleep(0.4)
            st.write("🧠 Fitting Linear Regression model...")
            time.sleep(0.4)
            st.write("📊 Computing R² and MAE validation metrics...")
            time.sleep(0.4)
            st.session_state.workflow = run_workflow(training_df, scenario, feature_override=custom_features)
            st.session_state.last_action = "Model trained: metrics, actual-vs-predicted chart, and downstream prediction were refreshed."
            status.update(label="Training complete!", state="complete", expanded=False)
        st.rerun()

    metric_rows = pd.DataFrame(model_result["metrics"]).T.reset_index(names="Model")
    st.subheader("Model Metrics")
    st.dataframe(metric_rows, use_container_width=True, hide_index=True)
    st.plotly_chart(actual_vs_predicted_chart(model_result["test_frame"], chart_height), use_container_width=True)

elif page == PAGES[2]:
    st.markdown('<div class="cg-eyebrow">Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="cg-section-title">Send the ML forecast into the allocation solver.</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="cg-info-panel">
          <strong>What does Run Optimization do?</strong>
          <p>It takes the predicted energy value shown below and passes it into PuLP as the supply constraint. Then it allocates energy across civic loads to maximize priority-weighted service while protecting the configured critical minimum for {escape(CRITICAL_LABEL)}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("ML Output Passed Into LP")
    st.markdown(
        f"""
        <div class="cg-card">
          <h3>Predicted energy RHS</h3>
          <div class="cg-stat-number">{predicted_energy:.3f} kWh</div>
          <p>This value is the right-hand side of the LP supply constraint: total allocated energy must be less than or equal to the forecast.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    editable = pd.DataFrame(
        [{"load": name, **details} for name, details in loads.items()]
    )
    st.subheader("Judge-adjustable load settings")
    st.caption("Edit demand, priority, or critical minimums, then run the solver again. Demand and minimum values are in kWh.")
    edited = st.data_editor(editable, use_container_width=True, num_rows="fixed")
    edited_loads = {
        row["load"]: {
            "demand": float(row["demand"]),
            "priority": float(row["priority"]),
            "min_required": float(row["min_required"]),
            "label": str(row.get("label", row["load"])),
        }
        for _, row in edited.iterrows()
    }
    load_errors = []
    for load_name, details in edited_loads.items():
        label = details.get("label", load_name)
        if details["demand"] < 0:
            load_errors.append(f"{label} demand cannot be negative.")
        if details["priority"] < 0:
            load_errors.append(f"{label} priority cannot be negative.")
        if details["min_required"] < 0:
            load_errors.append(f"{label} minimum cannot be negative.")
        if details["min_required"] > details["demand"]:
            load_errors.append(f"{label} minimum cannot exceed demand.")
    if load_errors:
        st.error(" ".join(load_errors))

    if st.button("Run Optimization", use_container_width=True, disabled=bool(load_errors)):
        import time
        with st.status("Solving Microgrid LP...", expanded=True) as status:
            st.write("⚡ Fetching load demands and priority bounds...")
            time.sleep(0.4)
            st.write("🔧 Formulating constraints in PuLP...")
            time.sleep(0.4)
            st.write("🚀 Running CBC solver engine...")
            time.sleep(0.4)
            st.session_state.workflow = run_workflow(training_df, scenario, edited_loads, custom_features)
            st.session_state.last_action = "Optimization complete: edited load settings were solved with the current predicted energy."
            status.update(label="Optimization solver complete!", state="complete", expanded=False)
        st.rerun()

    status_class = "badge" if optimized["status"] == "Optimal" else "badge err"
    st.markdown(f'<span class="{status_class}">{optimized["status"]}</span>', unsafe_allow_html=True)
    if optimized["status"] != "Optimal":
        st.warning(f"Even the {CRITICAL_LABEL} minimum cannot be met with the predicted energy. The app handled the infeasible case without crashing.")
    else:
        critical_allocation = optimized["allocation"].get(CRITICAL_LOAD, 0)
        if critical_allocation >= loads[CRITICAL_LOAD]["min_required"]:
            st.success(f"{CRITICAL_LABEL} is fully protected before lower-priority loads receive scarce energy.")
    result_rows = pd.DataFrame(
        [
            {
                "Load": loads[name].get("label", name.replace("_", " ").title()),
                "Demand kWh": loads[name]["demand"],
                "Optimized kWh": value,
                "% of demand": round(value / loads[name]["demand"] * 100, 1) if loads[name]["demand"] else 0,
            }
            for name, value in optimized["allocation"].items()
        ]
    )
    st.dataframe(result_rows, use_container_width=True, hide_index=True)
    st.plotly_chart(allocation_chart(loads, optimized["allocation"], "Optimized Allocation", chart_height), use_container_width=True)

elif page == PAGES[3]:
    st.markdown('<div class="cg-eyebrow">Results & Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="cg-section-title">Compare optimized allocation against equal-cut baseline.</div>', unsafe_allow_html=True)
    if not st.session_state.get("show_results", True):
        st.warning("⚠️ No active simulation results. Go to the Home page, adjust inputs, and click 'Run Complete Demo' first!")
    else:
        st.markdown(
            """
            <div class="cg-info-panel">
              <strong>What is this page proving?</strong>
              <p>It compares the optimized allocation against a naive equal-cut baseline using three judge-friendly metrics: critical minimum met, priority-weighted score, and number of loads fully satisfied.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        columns = st.columns(3)
        for column, (label, values) in zip(columns, metrics.items()):
            delta = values["optimized"] - values["baseline"]
            column.metric(label, f"{values['optimized']:.1f}", delta=f"{delta:.1f} vs equal-cut")
        st.plotly_chart(comparison_chart(loads, optimized["allocation"], baseline, chart_height), use_container_width=True)
        st.download_button(
            "Download allocation CSV",
            data=allocation_to_csv(predicted_energy, loads, optimized["allocation"], baseline),
            file_name=CONFIG["app"]["download_filename"],
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(
            '<div class="cg-helper">Download allocation CSV exports the current optimized and baseline allocations so the result can be inspected outside the app.</div>',
            unsafe_allow_html=True,
        )

else:
    st.markdown('<div class="cg-eyebrow">Mathematics & About</div>', unsafe_allow_html=True)
    st.markdown('<div class="cg-section-title">The formulas behind each decision.</div>', unsafe_allow_html=True)
    st.subheader("Mathematics Behind the Decision")
    st.latex(r"\hat{E}=\beta_0+\beta_1S+\beta_2F+\beta_3T+\beta_4W")
    st.write("S is solar irradiance, F is footfall, T is temperature, and W marks weekend hours.")
    st.latex(r"\max \sum_{j \in L} w_j x_j")
    st.latex(r"\sum_{j \in L} x_j \leq \hat{E}, \quad 0 \leq x_j \leq d_j, \quad x_{critical} \geq m_{critical}")
    st.write(
        f"For the selected scenario, the trained model predicts {predicted_energy:.2f} kWh. That exact value becomes the supply constraint RHS in PuLP."
    )
    st.subheader("Real-World Production Architecture")
    st.markdown(
        """
        <div class="cg-info-panel" style="margin-bottom: 20px;">
          <strong>🔌 Production Deployment Architecture</strong>
          <p>In a live smart city deployment, CivicGrid AI operates as an automated <b>Active Load Management (ALM)</b> system:
          <ol>
            <li><b>Telemetry Ingestion:</b> IoT sensors (pyranometers, footfall cameras, battery telemetry) push live streams into a time-series database (e.g. InfluxDB).</li>
            <li><b>Rolling Forecasts:</b> A background microservice triggers predictions hourly using the pre-trained ML model.</li>
            <li><b>Constraint Optimization:</b> The predicted supply limit (RHS) is sent directly to the solver engine to determine optimal switch states.</li>
            <li><b>Hardware SCADA Dispatch:</b> The solved allocations are translated into PLC commands (Modbus/BACnet) to physically throttle and shed loads in real time, keeping Emergency Lighting fully protected.</li>
          </ol>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Impact, Limitations, Future Scope")
    st.write(
        "The prototype is synthetic and offline by design. Future work can add calibrated sensor data, weather forecasts, richer load classes, and municipal policy constraints."
    )
