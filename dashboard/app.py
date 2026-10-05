import os
import sys
import json
import yaml
import time
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.cells import CellNetwork
from simulator.routes import RouteGenerator
from simulator.mobility import MobilitySimulator
from simulator.radio import RadioEnvironment
from features.feature_engineering import FeaturePipeline
from models.next_cell_model import NextCellPredictor
from models.weakness_model import WeaknessPredictor
from models.inference import MobilityInferenceEngine
from decision.uncertainty import UncertaintyEstimator
from decision.divergence import RouteDivergenceDetector
from decision.policy import MobilityPolicyEngine, MobilityState
from storage.logger import EventLogger


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="YellowSense | Mobility Intelligence Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session Theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# -----------------------------------------------------------------------------
# 2. Design Token System (Light & Dark)
# -----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "mode": "dark",
        "bg_app": "#0B1120",         # Deep midnight navy canvas
        "bg_sidebar": "#070D1A",     # Seamless deeper sidebar
        "bg_card": "#131E35",        # Primary surface
        "bg_card_inner": "#0E172B",  # Inset surface / container
        "bg_card_alt": "#1E293B",    # Subtle elevated surface
        "bg_input": "#0F172A",
        "border": "#22314E",
        "border_subtle": "#1B273F",
        "text_primary": "#F8FAFC",   # Crisp soft white
        "text_secondary": "#94A3B8", # Muted slate
        "text_muted": "#64748B",     # Quiet caption
        "accent": "#F59E0B",         # YellowSense Gold
        "accent_hover": "#FBBF24",
        "accent_bg": "rgba(245, 158, 11, 0.12)",
        "accent_border": "rgba(245, 158, 11, 0.40)",
        "accent_text": "#FDE68A",
        "blue": "#3B82F6",
        "blue_bg": "rgba(59, 130, 246, 0.12)",
        "blue_border": "rgba(59, 130, 246, 0.40)",
        "blue_text": "#93C5FD",
        "green": "#10B981",
        "green_bg": "rgba(16, 185, 129, 0.12)",
        "green_border": "rgba(16, 185, 129, 0.40)",
        "green_text": "#6EE7B7",
        "red": "#EF4444",
        "red_bg": "rgba(239, 68, 68, 0.15)",
        "red_border": "rgba(239, 68, 68, 0.50)",
        "red_text": "#FCA5A5",
        "orange": "#FB923C",
        "purple": "#A855F7",
        "plotly_template": "plotly_dark",
        "plotly_paper": "#131E35",
        "plotly_plot": "#131E35",
        "plotly_grid": "#1E293B",
        "map_bg": "#0D1527",
    },
    "light": {
        "mode": "light",
        "bg_app": "#F8FAFC",         # Warm off-white canvas
        "bg_sidebar": "#F1F5F9",     # Cool neutral light sidebar
        "bg_card": "#FFFFFF",        # Crisp white surfaces
        "bg_card_inner": "#F8FAFC",  # Inset surface
        "bg_card_alt": "#EDF2F7",
        "bg_input": "#FFFFFF",
        "border": "#E2E8F0",
        "border_subtle": "#CBD5E1",
        "text_primary": "#0F172A",   # Deep navy
        "text_secondary": "#475569", # Slate
        "text_muted": "#64748B",
        "accent": "#D97706",         # Enterprise YellowSense Amber
        "accent_hover": "#B45309",
        "accent_bg": "rgba(217, 119, 6, 0.08)",
        "accent_border": "rgba(217, 119, 6, 0.35)",
        "accent_text": "#92400E",
        "blue": "#2563EB",
        "blue_bg": "rgba(37, 99, 235, 0.08)",
        "blue_border": "rgba(37, 99, 235, 0.35)",
        "blue_text": "#1E40AF",
        "green": "#059669",
        "green_bg": "rgba(5, 150, 105, 0.08)",
        "green_border": "rgba(5, 150, 105, 0.35)",
        "green_text": "#065F46",
        "red": "#DC2626",
        "red_bg": "rgba(220, 38, 38, 0.08)",
        "red_border": "rgba(220, 38, 38, 0.35)",
        "red_text": "#991B1B",
        "orange": "#EA580C",
        "purple": "#9333EA",
        "plotly_template": "plotly_white",
        "plotly_paper": "#FFFFFF",
        "plotly_plot": "#FFFFFF",
        "plotly_grid": "#E2E8F0",
        "map_bg": "#F8FAFC",
    },
}

current_theme = THEMES[st.session_state.theme]


# -----------------------------------------------------------------------------
# 3. Logo Helper
# -----------------------------------------------------------------------------
def load_brand_logo(theme_mode: str) -> str:
    """Loads the official YellowSense logo as a base64 data URI."""
    logo_filename = "yellowsense_logo_dark.png" if theme_mode == "dark" else "yellowsense_logo_light.png"
    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", logo_filename)
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"
    return ""


# -----------------------------------------------------------------------------
# 4. Comprehensive Enterprise CSS Injection
# -----------------------------------------------------------------------------
def generate_css(t: dict) -> str:
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {t['bg_app']} !important;
        color: {t['text_primary']} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }}
    .stApp {{
        background-color: {t['bg_app']} !important;
    }}

    /* Global Header Chrome */
    header[data-testid="stHeader"] {{
        background-color: {t['bg_app']} !important;
        border-bottom: 1px solid {t['border']};
    }}

    /* Sidebar Theme Override */
    section[data-testid="stSidebar"] {{
        background-color: {t['bg_sidebar']} !important;
        border-right: 1px solid {t['border']} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: {t['text_primary']} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {t['text_primary']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.2px;
    }}

    /* Input & Select Controls */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {{
        background-color: {t['bg_input']} !important;
        color: {t['text_primary']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 6px !important;
    }}
    .stSelectbox label, .stSlider label, .stCheckbox label {{
        color: {t['text_secondary']} !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }}

    /* Premium Segmented Radio Control */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: inline-flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        background-color: {t['bg_card_inner']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 20px !important;
        padding: 3px !important;
        gap: 3px !important;
        align-items: center !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        margin: 0 !important;
        padding: 5px 14px !important;
        border-radius: 16px !important;
        background-color: transparent !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        transition: all 0.15s ease-in-out !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
        display: none !important; /* Hide default radio button circle */
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] p {{
        font-size: 11.5px !important;
        font-weight: 700 !important;
        color: {t['text_secondary']} !important;
        letter-spacing: 0.5px !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        background-color: {t['bg_card']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {{
        color: {t['text_primary']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        background-color: {t['bg_card']} !important;
        border: 1px solid {t['border']} !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, {'0.3' if t['mode'] == 'dark' else '0.06'}) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
        color: {t['accent']} !important;
        font-weight: 800 !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {t['bg_card_alt']} !important;
        color: {t['text_primary']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 7px 16px !important;
        transition: all 0.15s ease-in-out !important;
    }}
    .stButton > button:hover {{
        border-color: {t['accent']} !important;
        color: {t['accent']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {t['accent']} !important;
        color: #000000 !important;
        border: 1px solid {t['accent_hover']} !important;
        font-weight: 700 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {t['accent_hover']} !important;
        color: #000000 !important;
    }}

    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 2px solid {t['border']} !important;
        gap: 6px !important;
        padding-bottom: 2px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border: none !important;
        color: {t['text_secondary']} !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.3px !important;
        padding: 9px 16px !important;
        border-radius: 6px 6px 0 0 !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {t['text_primary']} !important;
        background-color: {t['bg_card']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {t['accent']} !important;
        border-bottom: 3px solid {t['accent']} !important;
        background-color: {t['bg_card']} !important;
        font-weight: 700 !important;
    }}

    /* Dataframe / Table */
    .stDataFrame {{
        background-color: {t['bg_card']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}

    /* Card Containers */
    .ys-card {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, {'0.2' if t['mode'] == 'dark' else '0.03'});
    }}

    /* Top Console Header Bar */
    .ys-header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 14px 0;
        border-bottom: 1px solid {t['border']};
        margin-bottom: 16px;
    }}
    .ys-header-left {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .ys-header-logo {{
        height: 52px;
        width: auto;
        object-fit: contain;
        display: block;
        margin-top: -2px; /* Lift logo upward for visual center balance */
    }}
    .ys-header-divider {{
        height: 42px;
        width: 1.5px;
        background-color: {t['border']};
    }}
    .ys-header-title {{
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.6px;
        color: {t['text_primary']};
        line-height: 1.15;
    }}
    .ys-header-sub {{
        font-size: 12px;
        color: {t['text_secondary']};
        margin-top: 3px;
        font-weight: 500;
    }}
    .ys-header-right {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    /* Mission Status Banner */
    .ys-mission-bar {{
        background-color: {t['bg_card_inner']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 8px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }}
    .ys-mission-left {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        color: {t['text_primary']};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .ys-mission-flow {{
        font-size: 11.5px;
        color: {t['text_secondary']};
        font-weight: 500;
    }}

    /* Live Status Badge */
    .ys-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .ys-status-live {{
        background-color: {t['green_bg']};
        color: {t['green']};
        border: 1px solid {t['green_border']};
    }}
    .ys-status-paused {{
        background-color: {t['accent_bg']};
        color: {t['accent']};
        border: 1px solid {t['accent_border']};
    }}
    .ys-status-ready {{
        background-color: {t['blue_bg']};
        color: {t['blue']};
        border: 1px solid {t['blue_border']};
    }}

    /* Connected Causal Strip KPI Card */
    .ys-kpi-card {{
        background-color: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 13px 15px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        box-shadow: 0 2px 6px rgba(0, 0, 0, {'0.2' if t['mode'] == 'dark' else '0.03'});
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .ys-kpi-card:hover {{
        border-color: {t['accent']};
    }}
    .ys-kpi-num {{
        font-size: 10px;
        font-weight: 800;
        color: {t['text_muted']};
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 2px;
    }}
    .ys-kpi-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {t['text_secondary']};
        margin-bottom: 4px;
    }}
    .ys-kpi-val {{
        font-size: 24px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 5px;
    }}
    .ys-kpi-sub {{
        font-size: 11.5px;
        color: {t['text_secondary']};
        font-weight: 500;
    }}

    /* State Progression Stepper */
    .ys-stepper {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {t['bg_card_inner']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 9px 16px;
        margin: 12px 0 14px 0;
    }}
    .ys-step-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }}
    .ys-step-arrow {{
        color: {t['text_muted']};
        font-size: 11px;
    }}

    /* Hero Divergence Alert Banner */
    .ys-hero-alert {{
        background: {'linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.25) 100%)' if t['mode'] == 'dark' else 'linear-gradient(135deg, rgba(254, 226, 226, 0.8) 0%, rgba(254, 202, 202, 0.5) 100%)'};
        border: 1.5px solid {t['red']};
        border-radius: 8px;
        padding: 15px 18px;
        margin: 14px 0;
        box-shadow: 0 4px 16px rgba(239, 68, 68, {'0.25' if t['mode'] == 'dark' else '0.08'});
    }}
    .ys-hero-title {{
        font-size: 14.5px;
        font-weight: 800;
        color: {t['red']};
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .ys-hero-msg {{
        font-size: 12.5px;
        color: {t['text_primary']};
        font-weight: 500;
        line-height: 1.4;
    }}

    /* Reason Explanatory Box */
    .ys-reason-box {{
        background-color: {t['bg_card_inner']};
        border-left: 3px solid {t['accent']};
        border-radius: 0 6px 6px 0;
        padding: 9px 13px;
        font-size: 12px;
        color: {t['text_primary']};
        line-height: 1.45;
        margin-top: 8px;
    }}

    /* Footer */
    .ys-footer {{
        border-top: 1px solid {t['border']};
        padding: 16px 0;
        margin-top: 32px;
        text-align: center;
        font-size: 11.5px;
        color: {t['text_muted']};
    }}
</style>
"""

st.markdown(generate_css(current_theme), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Engine & Model Resource Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_app_resources():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    network = CellNetwork(config.get("cells", {}))
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")

    if not os.path.exists(next_cell_path) or not os.path.exists(weakness_path):
        from scripts.train import main as train_main
        train_main()

    next_cell_model = NextCellPredictor.load(next_cell_path)
    weakness_model = WeaknessPredictor.load(weakness_path)
    feat_pipe = FeaturePipeline(cell_list=next_cell_model.cell_classes)
    inference_engine = MobilityInferenceEngine(next_cell_model, weakness_model, feat_pipe)

    uncertainty_estimator = UncertaintyEstimator()
    divergence_detector = RouteDivergenceDetector(network)
    policy_engine = MobilityPolicyEngine(
        divergence_detector,
        conf_high_th=config.get("policy", {}).get("confidence_high_threshold", 0.80),
        conf_med_th=config.get("policy", {}).get("confidence_medium_threshold", 0.60),
        weakness_risk_th=config.get("policy", {}).get("weakening_risk_threshold", 0.50),
    )
    logger = EventLogger()

    return {
        "config": config,
        "network": network,
        "inference_engine": inference_engine,
        "uncertainty_estimator": uncertainty_estimator,
        "policy_engine": policy_engine,
        "logger": logger,
    }


res = load_app_resources()
config = res["config"]
network = res["network"]
inference_engine = res["inference_engine"]
uncertainty_estimator = res["uncertainty_estimator"]
policy_engine = res["policy_engine"]
logger = res["logger"]


# -----------------------------------------------------------------------------
# 6. Session State Management
# -----------------------------------------------------------------------------
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "highway_exit"
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "mobility_sim" not in st.session_state:
    wps = RouteGenerator.generate_waypoints("highway_exit", seed=42)
    st.session_state.mobility_sim = MobilitySimulator(wps, dt=1.0)
if "radio_sim" not in st.session_state:
    st.session_state.radio_sim = RadioEnvironment(network, seed=42)
if "sim_history" not in st.session_state:
    st.session_state.sim_history = []
if "event_timeline" not in st.session_state:
    st.session_state.event_timeline = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = 0.25
if "auto_paused_reason" not in st.session_state:
    st.session_state.auto_paused_reason = None
if "paused_milestones_hit" not in st.session_state:
    st.session_state.paused_milestones_hit = set()


def reset_simulation(scenario_name: str):
    st.session_state.current_scenario = scenario_name
    st.session_state.step_idx = 0
    wps = RouteGenerator.generate_waypoints(scenario_name, seed=42)
    st.session_state.mobility_sim = MobilitySimulator(wps, dt=1.0)
    st.session_state.radio_sim = RadioEnvironment(network, seed=42)
    st.session_state.sim_history = []
    st.session_state.event_timeline = []
    st.session_state.is_running = False
    st.session_state.auto_paused_reason = None
    st.session_state.paused_milestones_hit.clear()
    policy_engine.reset()


# -----------------------------------------------------------------------------
# 7. Sidebar: Simulation Control Center
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 6px 0 14px 0; border-bottom: 1px solid {current_theme['border']}; margin-bottom: 14px;">
        <div style="font-size: 15px; font-weight: 800; color: {current_theme['accent']}; letter-spacing: 0.5px;">
            SIMULATION CONTROL
        </div>
        <div style="font-size: 11.5px; color: {current_theme['text_secondary']}; margin-top: 2px;">
            YellowSense Mobility Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### SCENARIO")
    scenario_options = {
        "highway_exit": "Highway Route Deviation (Primary Demo)",
        "highway_straight": "Highway Straight Corridor (A -> B -> E)",
        "urban_grid": "Urban Grid Multi-Turn",
        "rail_corridor": "High-Speed Rail Corridor (45 m/s)",
        "station_approach": "Station Deceleration Approach",
        "noisy_measurement": "Noisy Shadowing & Multipath",
        "sparse_telemetry": "Sparse / Jittered Telemetry",
    }

    selected_sc = st.selectbox(
        "Trajectory Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0,
        label_visibility="collapsed",
    )

    if selected_sc != st.session_state.current_scenario:
        reset_simulation(selected_sc)
        st.rerun()

    st.markdown("#### EXECUTION")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.is_running = True
            st.session_state.auto_paused_reason = None
    with col_b2:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.is_running = False

    col_b3, col_b4 = st.columns(2)
    with col_b3:
        step_clicked = st.button("STEP", use_container_width=True)
        if step_clicked:
            st.session_state.auto_paused_reason = None
    with col_b4:
        if st.button("RESET", use_container_width=True):
            reset_simulation(selected_sc)
            st.rerun()

    st.markdown("#### PLAYBACK & MILESTONES")
    auto_pause_enabled = st.checkbox(
        "Auto-Pause at Key Milestones",
        value=True,
        help="Pauses playback automatically upon reaching Proactive Candidate, Route Divergence, or Updated Candidate.",
    )
    st.session_state.playback_speed = st.slider("Step Interval (s):", 0.05, 1.0, 0.25, 0.05)

    st.markdown(f"""
    <div style="padding-top: 14px; border-top: 1px solid {current_theme['border']}; margin-top: 14px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: {current_theme['text_secondary']}; margin-bottom: 8px;">
            ENGINE SPECIFICATIONS
        </div>
        <div style="font-size: 11.5px; color: {current_theme['text_secondary']}; line-height: 1.6;">
            <div>• Model: <span style="color: {current_theme['text_primary']}; font-family: 'JetBrains Mono', monospace;">XGB-Calibrated</span></div>
            <div>• Horizon: <span style="color: {current_theme['text_primary']}; font-family: 'JetBrains Mono', monospace;">4.0s (4 steps)</span></div>
            <div>• High Conf Gate: <span style="color: {current_theme['text_primary']}; font-family: 'JetBrains Mono', monospace;">C ≥ 0.80</span></div>
            <div>• Weakness Gate: <span style="color: {current_theme['text_primary']}; font-family: 'JetBrains Mono', monospace;">W ≥ 0.50</span></div>
            <div>• Entropy Gate: <span style="color: {current_theme['text_primary']}; font-family: 'JetBrains Mono', monospace;">H ≤ 0.65</span></div>
            <div>• Operating Mode: <span style="color: {current_theme['accent']}; font-weight: 600;">Advisory POC</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Advance Simulation Step Function
# -----------------------------------------------------------------------------
def advance_simulation_step():
    mobility = st.session_state.mobility_sim
    radio = st.session_state.radio_sim
    step = st.session_state.step_idx

    noise_mult = 3.0 if st.session_state.current_scenario == "noisy_measurement" else 1.0
    kin = mobility.step()
    rad = radio.compute_telemetry(kin["x"], kin["y"], dt=1.0, noise_multiplier=noise_mult)

    telemetry = {
        "trajectory_id": f"gui_{st.session_state.current_scenario}",
        "ue_id": "UE_LIVE_01",
        "scenario": st.session_state.current_scenario,
        "step_index": step,
        "timestamp": round(step * 1.0, 1),
        "x": kin["x"],
        "y": kin["y"],
        "speed_mps": kin["speed_mps"],
        "heading_deg": kin["heading_deg"],
        "heading_change_deg": kin["heading_change_deg"],
        **rad,
    }

    # Run Real AI Pipeline (XGBoost -> Platt Calibration -> Entropy -> Policy)
    inf_res = inference_engine.run_inference(telemetry, recent_history=st.session_state.sim_history)
    unc_res = uncertainty_estimator.evaluate(inf_res["probabilities"])
    decision = policy_engine.evaluate(telemetry, inf_res, unc_res, step_index=step)

    # Check for Auto-Pause Milestones
    if auto_pause_enabled and st.session_state.is_running:
        if decision.action == "PROACTIVE_CANDIDATE" and "PROACTIVE" not in st.session_state.paused_milestones_hit:
            st.session_state.is_running = False
            st.session_state.auto_paused_reason = f"PROACTIVE CANDIDATE ({decision.candidate_cell}) IDENTIFIED"
            st.session_state.paused_milestones_hit.add("PROACTIVE")
        elif decision.divergence_detected and "DIVERGENCE" not in st.session_state.paused_milestones_hit:
            st.session_state.is_running = False
            st.session_state.auto_paused_reason = "ROUTE DIVERGENCE DETECTED (PREDICTION INVALIDATED)"
            st.session_state.paused_milestones_hit.add("DIVERGENCE")
        elif decision.state == MobilityState.UPDATED_CANDIDATE and "UPDATED" not in st.session_state.paused_milestones_hit:
            st.session_state.is_running = False
            st.session_state.auto_paused_reason = f"UPDATED CANDIDATE ({decision.candidate_cell}) SELECTED"
            st.session_state.paused_milestones_hit.add("UPDATED")

    # Record event
    event = {
        "step": step,
        "time": round(step * 1.0, 1),
        "x": kin["x"],
        "y": kin["y"],
        "speed": kin["speed_mps"],
        "heading": kin["heading_deg"],
        "serving_cell": rad["current_cell"],
        "serving_rsrp": rad["current_rsrp_dbm"],
        "serving_sinr": rad["current_sinr_db"],
        "predicted_cell": unc_res.predicted_cell,
        "confidence": unc_res.confidence,
        "confidence_pct": unc_res.confidence_percentage,
        "entropy": unc_res.entropy,
        "normalized_entropy": unc_res.normalized_entropy,
        "uncertainty_level": unc_res.uncertainty_level,
        "weakness_risk": inf_res["weakness_risk"],
        "action": decision.action,
        "candidate_cell": decision.candidate_cell,
        "state": decision.state.value,
        "reason": decision.reason,
        "divergence": decision.divergence_detected,
        "probabilities": inf_res["probabilities"],
        "telemetry_raw": telemetry,
        "latency_ms": inf_res["latency_ms"],
        "is_finished": kin["is_finished"],
    }

    st.session_state.sim_history.append(telemetry)
    st.session_state.event_timeline.append(event)
    st.session_state.step_idx += 1

    if kin["is_finished"]:
        st.session_state.is_running = False

    return event


if step_clicked or st.session_state.is_running:
    latest_event = advance_simulation_step()
elif st.session_state.event_timeline:
    latest_event = st.session_state.event_timeline[-1]
else:
    latest_event = advance_simulation_step()


# -----------------------------------------------------------------------------
# 9. Top Application Header with Official Brand Logo & Segmented Theme Control
# -----------------------------------------------------------------------------
logo_data_uri = load_brand_logo(st.session_state.theme)

col_header_left, col_header_right = st.columns([0.68, 0.32])

with col_header_left:
    logo_img_tag = f'<img src="{logo_data_uri}" class="ys-header-logo" alt="YellowSense" />' if logo_data_uri else f'<span style="font-size: 20px; font-weight: 900; letter-spacing: 1px; color: {current_theme["accent"]};">YELLOWSENSE</span>'
    st.markdown(f"""
    <div class="ys-header-left">
        {logo_img_tag}
        <div class="ys-header-divider"></div>
        <div>
            <div class="ys-header-title">
                AI-NATIVE MOBILITY INTELLIGENCE
            </div>
            <div class="ys-header-sub">
                Uncertainty-Aware Prediction &amp; Adaptive Decisions
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_header_right:
    col_status, col_theme = st.columns([0.52, 0.48])
    with col_status:
        if st.session_state.is_running:
            status_html = f'<div class="ys-status-pill ys-status-live" style="width: 100%; text-align: center; justify-content: center; height: 36px; display: flex; align-items: center;">● LIVE SIMULATION</div>'
        elif st.session_state.auto_paused_reason:
            status_html = f'<div class="ys-status-pill ys-status-paused" style="width: 100%; text-align: center; justify-content: center; height: 36px; display: flex; align-items: center;">● MILESTONE PAUSE</div>'
        else:
            status_html = f'<div class="ys-status-pill ys-status-ready" style="width: 100%; text-align: center; justify-content: center; height: 36px; display: flex; align-items: center;">● READY</div>'
        st.markdown(status_html, unsafe_allow_html=True)

    with col_theme:
        theme_choice = st.radio(
            "Theme",
            options=["light", "dark"],
            format_func=lambda x: "☀ DAY" if x == "light" else "◐ NIGHT",
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

# Mission Status / Concept Bar
st.markdown(f"""
<div class="ys-mission-bar">
    <div class="ys-mission-left">
        <span style="color: {current_theme['accent']}; font-weight: 800;">SCENARIO:</span>
        <span>{scenario_options.get(st.session_state.current_scenario, 'Highway Route Deviation')}</span>
        <span style="color: {current_theme['border']}; margin: 0 4px;">|</span>
        <span style="color: {current_theme['green'] if st.session_state.is_running else current_theme['blue']}; font-weight: 700;">
            {'● SIMULATION ACTIVE' if st.session_state.is_running else '● READY FOR PLAYBACK'}
        </span>
    </div>
    <div class="ys-mission-flow">
        Predict &nbsp;→&nbsp; Assess &nbsp;→&nbsp; Decide &nbsp;→&nbsp; Observe &nbsp;→&nbsp; Adapt
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 10. Top Navigation Tabs
# -----------------------------------------------------------------------------
tab_live, tab_telemetry, tab_eval, tab_arch = st.tabs([
    "LIVE INTELLIGENCE",
    "BELIEF & TELEMETRY",
    "EVALUATION & BENCHMARKS",
    "ARCHITECTURE & ROADMAP",
])


# =============================================================================
# TAB 1: LIVE INTELLIGENCE
# =============================================================================
with tab_live:
    # Milestone Alert Banner if Auto-Paused
    if st.session_state.auto_paused_reason and not st.session_state.is_running:
        st.markdown(f"""
        <div style="background-color: {current_theme['accent_bg']}; border: 1px solid {current_theme['accent_border']}; border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 12.5px; font-weight: 700; color: {current_theme['accent_text'] if current_theme['mode'] == 'dark' else current_theme['accent']};">
                ⏸ AUTO-PAUSED AT MILESTONE: {st.session_state.auto_paused_reason}
            </div>
            <div style="font-size: 11.5px; color: {current_theme['text_secondary']};">
                Press <b>START</b> to resume or <b>STEP</b> for granular inspection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5-Card Causal KPI Strip
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

    with col_kpi1:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-num">01 CURRENT</div>
            <div class="ys-kpi-label">Serving Cell</div>
            <div class="ys-kpi-val" style="color: {current_theme['blue']};">{latest_event['serving_cell']}</div>
            <div class="ys-kpi-sub">RSRP: <b>{latest_event['serving_rsrp']:.1f} dBm</b><br>SINR: <b>{latest_event['serving_sinr']:.1f} dB</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        pred_color = current_theme['green'] if latest_event["confidence"] >= 0.80 else (current_theme['accent'] if latest_event["confidence"] >= 0.60 else current_theme['red'])
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-num">02 PREDICTED FUTURE</div>
            <div class="ys-kpi-label">Next Cell</div>
            <div class="ys-kpi-val" style="color: {pred_color};">{latest_event['predicted_cell']}</div>
            <div class="ys-kpi-sub">Lookahead: <b>4.0s (4 steps)</b><br>Target Base Station</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-num">03 TRUST / CALIBRATION</div>
            <div class="ys-kpi-label">Confidence</div>
            <div class="ys-kpi-val" style="color: {pred_color};">{latest_event['confidence_pct']:.1f}%</div>
            <div class="ys-kpi-sub">Uncertainty: <b style="color: {pred_color};">{latest_event['uncertainty_level']}</b><br>Norm Entropy: <b>{latest_event['normalized_entropy']:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        weak_color = current_theme['red'] if latest_event["weakness_risk"] >= 0.50 else current_theme['green']
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-num">04 RISK ASSESSMENT</div>
            <div class="ys-kpi-label">Weakening Risk</div>
            <div class="ys-kpi-val" style="color: {weak_color};">{latest_event['weakness_risk'] * 100:.1f}%</div>
            <div class="ys-kpi-sub">Serving Degradation: <b style="color: {weak_color};">{'ELEVATED' if latest_event['weakness_risk'] >= 0.50 else 'STABLE'}</b><br>Threshold Gate: <b>50.0%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi5:
        act_color = current_theme['green'] if latest_event["action"] == "PROACTIVE_CANDIDATE" else (current_theme['red'] if latest_event["divergence"] else current_theme['blue'])
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-num">05 ACTION</div>
            <div class="ys-kpi-label">Policy Decision</div>
            <div class="ys-kpi-val" style="font-size: 17px; color: {act_color}; line-height: 1.25; padding-top: 3px;">{latest_event['action']}</div>
            <div class="ys-kpi-sub">Target Candidate: <b>{latest_event.get('candidate_cell') or 'NONE'}</b><br>State: <b>{latest_event['state']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Dynamic AI State Progression Stepper
    is_div = latest_event["divergence"]
    curr_state = latest_event["state"]

    predict_done = "✓" if latest_event["step"] >= 1 else "..."
    assess_done = "✓" if latest_event["step"] >= 1 else "..."
    decide_done = "✓" if latest_event["action"] != "NONE" else "..."
    observe_status = "⚠ DIVERGED" if is_div else "✓ ALIGNED"
    adapt_status = f"→ ADAPTING ({latest_event.get('candidate_cell', 'Cell_C')})" if is_div else ("✓ PREPARED" if latest_event["action"] == "PROACTIVE_CANDIDATE" else "MONITORING")

    st.markdown(f"""
    <div class="ys-stepper">
        <div class="ys-step-item" style="color: {current_theme['blue']};">
            <span>PREDICT</span> <span style="font-weight: 800;">{predict_done}</span>
        </div>
        <div class="ys-step-arrow">→</div>
        <div class="ys-step-item" style="color: {current_theme['blue']};">
            <span>ASSESS</span> <span style="font-weight: 800;">{assess_done}</span>
        </div>
        <div class="ys-step-arrow">→</div>
        <div class="ys-step-item" style="color: {current_theme['green'] if not is_div else current_theme['accent']};">
            <span>DECIDE</span> <span style="font-weight: 800;">{decide_done}</span>
        </div>
        <div class="ys-step-arrow">→</div>
        <div class="ys-step-item" style="color: {current_theme['red'] if is_div else current_theme['green']};">
            <span>OBSERVE</span> <span style="font-weight: 800;">{observe_status}</span>
        </div>
        <div class="ys-step-arrow">→</div>
        <div class="ys-step-item" style="color: {current_theme['accent'] if is_div else current_theme['green']};">
            <span>ADAPT</span> <span style="font-weight: 800;">{adapt_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Event Area (Route Divergence Banner or Structured Policy Card)
    if is_div:
        st.markdown(f"""
        <div class="ys-hero-alert">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div class="ys-hero-title">
                    <span>ROUTE DIVERGENCE DETECTED</span>
                    <span style="font-size: 11.5px; padding: 2px 8px; border-radius: 4px; background-color: {current_theme['red_bg']}; border: 1px solid {current_theme['red_border']}; color: {current_theme['red']};">
                        PREDICTION NO LONGER RELIABLE
                    </span>
                </div>
                <div style="font-size: 12px; font-weight: 800; color: {current_theme['accent']};">
                    RE-EVALUATING &nbsp;→&nbsp; {latest_event.get('candidate_cell', 'Cell_C')} EMERGING
                </div>
            </div>
            <div class="ys-hero-msg">
                Observed UE trajectory is no longer consistent with the predicted Cell_B corridor. The system immediately suppressed stale preparation and initiated adaptive re-evaluation.
            </div>
            <div class="ys-reason-box" style="border-left-color: {current_theme['red']};">
                <b>Policy Engine Rationale:</b> {latest_event['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        policy_badge_color = current_theme['green'] if latest_event["action"] == "PROACTIVE_CANDIDATE" else current_theme['blue']
        st.markdown(f"""
        <div class="ys-card" style="margin-bottom: 14px; padding: 13px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <div style="font-size: 13px; font-weight: 700; color: {current_theme['text_primary']}; display: flex; align-items: center; gap: 8px;">
                    <span>AI POLICY ADVISORY:</span>
                    <span style="color: {policy_badge_color}; font-weight: 800;">{latest_event['action']}</span>
                </div>
                <div style="font-size: 11.5px; color: {current_theme['text_secondary']};">
                    Step: <b>{latest_event['step']}</b> | Time: <b>{latest_event['time']:.1f}s</b> | Speed: <b>{latest_event['speed']:.1f} m/s</b> ({latest_event['heading']:.0f}°) | Inference: <b>{latest_event['latency_ms']:.1f} ms</b>
                </div>
            </div>
            <div class="ys-reason-box">
                <b>Decision Rationale:</b> {latest_event['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main Visual Area: 2D Topology Map (65%) & Probability Distribution (35%)
    col_map, col_probs = st.columns([1.25, 0.75])

    with col_map:
        st.markdown(f"""
        <div style="font-size: 13.5px; font-weight: 700; color: {current_theme['text_primary']}; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span>5G Multi-Cell Topology &amp; Live Mobility Trajectory</span>
            <span style="font-size: 11px; color: {current_theme['text_secondary']}; font-weight: 500;">Hexagonal Microcells (r=650m)</span>
        </div>
        """, unsafe_allow_html=True)

        fig_map = go.Figure()

        # Add Cell Towers & Coverage Contours
        for cid, cell in network.cells.items():
            theta = np.linspace(0, 2 * np.pi, 60)
            cx = cell.x + 650 * np.cos(theta)
            cy = cell.y + 650 * np.sin(theta)

            is_serving = (cid == latest_event["serving_cell"])
            is_predicted = (cid == latest_event["predicted_cell"])
            is_updated_cand = (cid == latest_event.get("candidate_cell") and latest_event["divergence"])

            if is_serving:
                fill_color = "rgba(59, 130, 246, 0.15)" if current_theme['mode'] == 'dark' else "rgba(37, 99, 235, 0.12)"
                border_color = current_theme['blue']
                border_width = 2.5
                dash_style = "solid"
            elif is_updated_cand:
                fill_color = "rgba(245, 158, 11, 0.20)" if current_theme['mode'] == 'dark' else "rgba(217, 119, 6, 0.15)"
                border_color = current_theme['accent']
                border_width = 3.0
                dash_style = "solid"
            elif is_predicted:
                fill_color = "rgba(16, 185, 129, 0.15)" if current_theme['mode'] == 'dark' else "rgba(5, 150, 105, 0.12)"
                border_color = current_theme['green']
                border_width = 2.5
                dash_style = "dot"
            else:
                fill_color = "rgba(148, 163, 184, 0.03)"
                border_color = current_theme['border']
                border_width = 1.0
                dash_style = "dot"

            fig_map.add_trace(go.Scatter(
                x=cx, y=cy,
                mode="lines",
                line=dict(color=border_color, width=border_width, dash=dash_style),
                fill="toself",
                fillcolor=fill_color,
                hoverinfo="skip",
                showlegend=False,
            ))

            # Tower Marker
            fig_map.add_trace(go.Scatter(
                x=[cell.x], y=[cell.y],
                mode="markers+text",
                marker=dict(size=14, color=cell.color, symbol="triangle-up", line=dict(color=current_theme['text_primary'], width=1.5)),
                text=[f"<b>{cell.id}</b>"],
                textposition="top center",
                textfont=dict(color=current_theme['text_primary'], size=11, family="Inter"),
                name=cell.label,
                hoverinfo="text",
                hovertext=f"<b>{cell.label}</b> ({cell.id})<br>Position: ({cell.x:.0f}m, {cell.y:.0f}m)<br>Frequency: {cell.frequency_ghz} GHz",
                showlegend=False,
            ))

        # Expected Route Marking (for highway_exit)
        if st.session_state.current_scenario == "highway_exit":
            fig_map.add_trace(go.Scatter(
                x=[0, 2000], y=[0, 0],
                mode="lines",
                line=dict(color=current_theme['text_muted'], width=2, dash="dash"),
                name="Expected Corridor (A → B)",
                hoverinfo="skip",
            ))

        # Trajectory History Trail
        if st.session_state.event_timeline:
            hist_x = [e["x"] for e in st.session_state.event_timeline]
            hist_y = [e["y"] for e in st.session_state.event_timeline]

            fig_map.add_trace(go.Scatter(
                x=hist_x, y=hist_y,
                mode="lines+markers",
                line=dict(color=current_theme['accent'], width=3.5),
                marker=dict(size=4, color=current_theme['accent']),
                name="Actual UE Trajectory",
            ))

        # Current UE Position
        ue_marker_color = current_theme['red'] if latest_event["divergence"] else current_theme['green']
        fig_map.add_trace(go.Scatter(
            x=[latest_event["x"]], y=[latest_event["y"]],
            mode="markers+text",
            marker=dict(size=18, color=ue_marker_color, symbol="circle", line=dict(color="#FFFFFF", width=2.5)),
            text=["<b>UE</b>"],
            textposition="bottom center",
            textfont=dict(color=current_theme['text_primary'], size=11, family="Inter"),
            name="Current UE Position",
        ))

        # Vector Arrow to Predicted Target
        cand_cell = network.get_cell(latest_event["predicted_cell"])
        fig_map.add_annotation(
            x=cand_cell.x, y=cand_cell.y,
            ax=latest_event["x"], ay=latest_event["y"],
            xref="x", yref="y", axref="x", ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2.5,
            arrowcolor=current_theme['green'] if not latest_event["divergence"] else current_theme['red'],
        )

        fig_map.update_layout(
            template=current_theme['plotly_template'],
            paper_bgcolor=current_theme['plotly_paper'],
            plot_bgcolor=current_theme['plotly_plot'],
            margin=dict(l=15, r=15, t=15, b=15),
            xaxis=dict(
                title="X Coordinate (Meters)",
                range=[-500, 4300],
                gridcolor=current_theme['plotly_grid'],
                title_font=dict(color=current_theme['text_secondary'], size=11),
                tickfont=dict(color=current_theme['text_secondary'], size=10),
            ),
            yaxis=dict(
                title="Y Coordinate (Meters)",
                range=[-500, 2300],
                gridcolor=current_theme['plotly_grid'],
                scaleanchor="x",
                scaleratio=1,
                title_font=dict(color=current_theme['text_secondary'], size=11),
                tickfont=dict(color=current_theme['text_secondary'], size=10),
            ),
            height=410,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(color=current_theme['text_secondary'], size=10.5),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    with col_probs:
        st.markdown(f"""
        <div style="font-size: 13.5px; font-weight: 700; color: {current_theme['text_primary']}; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span>AI Next-Cell Belief</span>
            <span style="font-size: 11px; color: {current_theme['text_secondary']}; font-weight: 500;">Calibrated Distribution</span>
        </div>
        """, unsafe_allow_html=True)

        probs = latest_event["probabilities"]
        cells_sorted = sorted(probs.keys())
        prob_vals = [probs[c] * 100 for c in cells_sorted]
        bar_colors = [
            current_theme['green'] if c == latest_event["predicted_cell"] else (
                current_theme['blue'] if c == latest_event["serving_cell"] else current_theme['border_subtle']
            )
            for c in cells_sorted
        ]

        fig_bar = go.Figure(go.Bar(
            x=prob_vals,
            y=cells_sorted,
            orientation="h",
            text=[f"{v:.1f}%" for v in prob_vals],
            textposition="auto",
            textfont=dict(color=current_theme['text_primary'] if current_theme['mode'] == 'dark' else "#000000", size=11, family="Inter"),
            marker_color=bar_colors,
        ))
        fig_bar.update_layout(
            template=current_theme['plotly_template'],
            paper_bgcolor=current_theme['plotly_paper'],
            plot_bgcolor=current_theme['plotly_plot'],
            margin=dict(l=15, r=15, t=15, b=15),
            xaxis=dict(
                title="Calibrated Posterior Probability (%)",
                range=[0, 100],
                gridcolor=current_theme['plotly_grid'],
                title_font=dict(color=current_theme['text_secondary'], size=11),
                tickfont=dict(color=current_theme['text_secondary'], size=10),
            ),
            yaxis=dict(
                title="Cell Candidate",
                autorange="reversed",
                title_font=dict(color=current_theme['text_secondary'], size=11),
                tickfont=dict(color=current_theme['text_secondary'], size=11),
            ),
            height=410,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # Bottom Event Audit Timeline
    st.markdown(f"""
    <div style="font-size: 13.5px; font-weight: 700; color: {current_theme['text_primary']}; margin: 18px 0 8px 0; display: flex; justify-content: space-between; align-items: center;">
        <span>Decision Event Audit Stream</span>
        <span style="font-size: 11.5px; color: {current_theme['text_secondary']};">Showing recent simulation steps</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.event_timeline:
        log_records = []
        for e in reversed(st.session_state.event_timeline[-10:]):
            status_tag = "DIVERGENCE" if e["divergence"] else ("PROACTIVE" if e["action"] == "PROACTIVE_CANDIDATE" else "MONITOR")
            log_records.append({
                "Step": e["step"],
                "Time (s)": f"{e['time']:.1f}",
                "Serving": e["serving_cell"],
                "RSRP (dBm)": f"{e['serving_rsrp']:.1f}",
                "Predicted": e["predicted_cell"],
                "Confidence": f"{e['confidence_pct']:.1f}%",
                "Uncertainty": e["uncertainty_level"],
                "Weakness": f"{e['weakness_risk'] * 100:.1f}%",
                "Policy Action": e["action"],
                "Status": status_tag,
            })
        st.dataframe(pd.DataFrame(log_records), use_container_width=True, hide_index=True)

    if st.session_state.is_running and not latest_event.get("is_finished", False):
        time.sleep(st.session_state.playback_speed)
        st.rerun()


# =============================================================================
# TAB 2: BELIEF & TELEMETRY
# =============================================================================
with tab_telemetry:
    st.markdown(f"""
    <div style="margin-bottom: 14px;">
        <div style="font-size: 15px; font-weight: 700; color: {current_theme['text_primary']};">
            AI Belief Dynamics &amp; RF Telemetry Trends
        </div>
        <div style="font-size: 12px; color: {current_theme['text_secondary']}; margin-top: 2px;">
            Dynamic adaptation of calibrated confidence, Shannon entropy uncertainty, and radio signal measurements across time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why This Decision Box
    st.markdown(f"""
    <div class="ys-card">
        <div style="font-size: 13px; font-weight: 700; color: {current_theme['accent']}; margin-bottom: 6px;">
            WHY THIS DECISION? (EXPLAINABILITY BREAKDOWN)
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; font-size: 12px; color: {current_theme['text_secondary']};">
            <div>• Target: <b style="color: {current_theme['text_primary']};">{latest_event['predicted_cell']}</b></div>
            <div>• Confidence: <b style="color: {current_theme['text_primary']};">{latest_event['confidence_pct']:.1f}%</b></div>
            <div>• Weakness Risk: <b style="color: {current_theme['text_primary']};">{latest_event['weakness_risk'] * 100:.1f}%</b></div>
            <div>• Trajectory: <b style="color: {current_theme['text_primary']};">{'Diverged' if latest_event['divergence'] else 'Corridor Aligned'}</b></div>
            <div>• Action: <b style="color: {current_theme['accent']};">{latest_event['action']}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.event_timeline) > 1:
        times = [e["time"] for e in st.session_state.event_timeline]
        confs = [e["confidence_pct"] for e in st.session_state.event_timeline]
        entropies = [e["normalized_entropy"] * 100 for e in st.session_state.event_timeline]
        weak_risks = [e["weakness_risk"] * 100 for e in st.session_state.event_timeline]

        serv_rsrps = [e["serving_rsrp"] for e in st.session_state.event_timeline]
        cand_b_rsrps = [e["telemetry_raw"].get("neighbor_Cell_B_rsrp", -120.0) for e in st.session_state.event_timeline]
        cand_c_rsrps = [e["telemetry_raw"].get("neighbor_Cell_C_rsrp", -120.0) for e in st.session_state.event_timeline]

        fig_hist = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                "AI Confidence (%), Normalized Shannon Entropy (%) & Weakening Risk (%)",
                "Serving vs Candidate Base Station RSRP Telemetry (dBm)"
            ),
        )

        # Subplot 1: Belief Metrics
        fig_hist.add_trace(go.Scatter(x=times, y=confs, name="Calibrated Confidence (%)", line=dict(color=current_theme['green'], width=2.5)), row=1, col=1)
        fig_hist.add_trace(go.Scatter(x=times, y=entropies, name="Normalized Entropy (%)", line=dict(color=current_theme['accent'], width=2, dash="dot")), row=1, col=1)
        fig_hist.add_trace(go.Scatter(x=times, y=weak_risks, name="Weakening Risk (%)", line=dict(color=current_theme['red'], width=1.8, dash="dash")), row=1, col=1)

        # Vertical line annotations for divergence events
        div_times = [e["time"] for e in st.session_state.event_timeline if e["divergence"]]
        for dt in div_times:
            fig_hist.add_vline(x=dt, line_width=1.5, line_dash="dash", line_color=current_theme['red'], annotation_text="DIVERGENCE", annotation_position="top right")

        # Subplot 2: RF Signals
        fig_hist.add_trace(go.Scatter(x=times, y=serv_rsrps, name="Serving Cell RSRP", line=dict(color=current_theme['blue'], width=2.5)), row=2, col=1)
        fig_hist.add_trace(go.Scatter(x=times, y=cand_b_rsrps, name="Cell_B RSRP", line=dict(color=current_theme['green'], width=1.5, dash="dot")), row=2, col=1)
        fig_hist.add_trace(go.Scatter(x=times, y=cand_c_rsrps, name="Cell_C RSRP", line=dict(color=current_theme['accent'], width=1.5, dash="dash")), row=2, col=1)

        fig_hist.update_layout(
            template=current_theme['plotly_template'],
            paper_bgcolor=current_theme['plotly_paper'],
            plot_bgcolor=current_theme['plotly_plot'],
            margin=dict(l=15, r=15, t=30, b=15),
            height=560,
            xaxis2=dict(title="Simulation Time (Seconds)", gridcolor=current_theme['plotly_grid'], title_font=dict(color=current_theme['text_secondary'])),
            yaxis=dict(title="Probability / Risk (%)", gridcolor=current_theme['plotly_grid'], range=[0, 105], title_font=dict(color=current_theme['text_secondary'])),
            yaxis2=dict(title="RSRP (dBm)", gridcolor=current_theme['plotly_grid'], title_font=dict(color=current_theme['text_secondary'])),
            legend=dict(font=dict(color=current_theme['text_secondary'], size=10.5), orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

        # Full Session Replay Download
        df_full_log = pd.DataFrame(st.session_state.event_timeline)
        csv_data = df_full_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Session Event Audit Log (CSV)",
            data=csv_data,
            file_name=f"yellowsense_audit_log_{st.session_state.current_scenario}.csv",
            mime="text/csv",
        )
    else:
        st.info("Advance simulation steps (click START or STEP) to view time-series belief dynamics.")


# =============================================================================
# TAB 3: EVALUATION & BENCHMARKS
# =============================================================================
with tab_eval:
    st.markdown(f"""
    <div style="margin-bottom: 14px;">
        <div style="font-size: 15px; font-weight: 700; color: {current_theme['text_primary']};">
            Simulation &amp; POC Evaluation Benchmarks
        </div>
        <div style="font-size: 12px; color: {current_theme['text_secondary']}; margin-top: 2px;">
            Held-out test split evaluation (4,465 unseen observations across independent synthetic trajectories). Labeled as <b>INTERNAL POC BENCHMARK</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-label">Top-1 Next-Cell Accuracy</div>
            <div class="ys-kpi-val" style="color: {current_theme['green']};">96.73%</div>
            <div class="ys-kpi-sub">+17.8% over reactive A3 standard</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-label">High-Conf Precision (C ≥ 0.80)</div>
            <div class="ys-kpi-val" style="color: {current_theme['green']};">97.73%</div>
            <div class="ys-kpi-sub">High-confidence quality gate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-label">Expected Calibration Error</div>
            <div class="ys-kpi-val" style="color: {current_theme['blue']};">0.0150</div>
            <div class="ys-kpi-sub">ECE via Platt Sigmoid scaling</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="ys-kpi-card">
            <div class="ys-kpi-label">Prediction Lead Time</div>
            <div class="ys-kpi-val" style="color: {current_theme['accent']};">4.0s</div>
            <div class="ys-kpi-sub">Lookahead horizon window</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 14px 0;'></div>", unsafe_allow_html=True)

    # Performance Breakdown Table
    st.markdown(f"""
    <div class="ys-card">
        <div style="font-size: 13px; font-weight: 700; color: {current_theme['text_primary']}; margin-bottom: 8px;">
            BENCHMARK PERFORMANCE SUMMARY (HELD-OUT TEST SPLIT — INTERNAL POC TARGETS)
        </div>
        <table style="width: 100%; font-size: 12px; border-collapse: collapse; color: {current_theme['text_primary']};">
            <thead>
                <tr style="border-bottom: 1px solid {current_theme['border']}; text-align: left; color: {current_theme['text_secondary']};">
                    <th style="padding: 8px 4px;">Metric Category</th>
                    <th style="padding: 8px 4px;">Evaluated Metric</th>
                    <th style="padding: 8px 4px;">Measured Value</th>
                    <th style="padding: 8px 4px;">Internal POC Target</th>
                    <th style="padding: 8px 4px;">Status</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid {current_theme['border']};">
                    <td style="padding: 8px 4px; font-weight: 600;">Prediction Quality</td>
                    <td style="padding: 8px 4px;">Next-Cell Top-1 Accuracy</td>
                    <td style="padding: 8px 4px; font-weight: 700; color: {current_theme['green']};">96.73%</td>
                    <td style="padding: 8px 4px; color: {current_theme['text_secondary']};">Target ≥ 85.0%</td>
                    <td style="padding: 8px 4px; color: {current_theme['green']}; font-weight: 600;">PASSED</td>
                </tr>
                <tr style="border-bottom: 1px solid {current_theme['border']};">
                    <td style="padding: 8px 4px; font-weight: 600;">Calibration Quality</td>
                    <td style="padding: 8px 4px;">Expected Calibration Error (ECE)</td>
                    <td style="padding: 8px 4px; font-weight: 700; color: {current_theme['green']};">0.0150</td>
                    <td style="padding: 8px 4px; color: {current_theme['text_secondary']};">Target ≤ 0.050</td>
                    <td style="padding: 8px 4px; color: {current_theme['green']}; font-weight: 600;">PASSED</td>
                </tr>
                <tr style="border-bottom: 1px solid {current_theme['border']};">
                    <td style="padding: 8px 4px; font-weight: 600;">Decision Safety</td>
                    <td style="padding: 8px 4px;">High-Conf Precision (C ≥ 0.80)</td>
                    <td style="padding: 8px 4px; font-weight: 700; color: {current_theme['green']};">97.73%</td>
                    <td style="padding: 8px 4px; color: {current_theme['text_secondary']};">Target ≥ 90.0%</td>
                    <td style="padding: 8px 4px; color: {current_theme['green']}; font-weight: 600;">PASSED</td>
                </tr>
                <tr>
                    <td style="padding: 8px 4px; font-weight: 600;">Inference Speed</td>
                    <td style="padding: 8px 4px;">Single-Step Inference Latency</td>
                    <td style="padding: 8px 4px; font-weight: 700; color: {current_theme['green']};">0.48 ms</td>
                    <td style="padding: 8px 4px; color: {current_theme['text_secondary']};">Target ≤ 10.0 ms</td>
                    <td style="padding: 8px 4px; color: {current_theme['green']}; font-weight: 600;">PASSED</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.markdown(f"<div style='font-size: 12.5px; font-weight: 700; color: {current_theme['text_primary']}; margin-bottom: 6px;'>Next-Cell Confusion Matrix</div>", unsafe_allow_html=True)
        cm_fig_path = os.path.join("artifacts", "figures", "confusion_matrix.png")
        if os.path.exists(cm_fig_path):
            st.image(cm_fig_path, use_container_width=True)
        else:
            st.info("Run `python scripts/evaluate.py` to generate figures.")

    with col_fig2:
        st.markdown(f"<div style='font-size: 12.5px; font-weight: 700; color: {current_theme['text_primary']}; margin-bottom: 6px;'>Reliability Diagram (Platt Calibration)</div>", unsafe_allow_html=True)
        calib_fig_path = os.path.join("artifacts", "figures", "calibration_curve.png")
        if os.path.exists(calib_fig_path):
            st.image(calib_fig_path, use_container_width=True)
        else:
            st.info("Run `python scripts/evaluate.py` to generate figures.")


# =============================================================================
# TAB 4: ARCHITECTURE & ROADMAP
# =============================================================================
with tab_arch:
    st.markdown(f"""
    <div style="margin-bottom: 14px;">
        <div style="font-size: 15px; font-weight: 700; color: {current_theme['text_primary']};">
            System Architecture, Claim Boundaries &amp; 5G Roadmap
        </div>
        <div style="font-size: 12px; color: {current_theme['text_secondary']}; margin-top: 2px;">
            End-to-end technical pipeline structure and phased carrier integration roadmap.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ys-card">
        <div style="font-size: 13px; font-weight: 700; color: {current_theme['accent']}; margin-bottom: 8px;">
            1. END-TO-END PIPELINE ARCHITECTURE (ACTIVE POC)
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.5; color: {current_theme['text_primary']};">
            TELEMETRY INGESTION (Kinematics, 3GPP Radio Measurements)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            CAUSAL FEATURE PIPELINE (Rolling Lags, Slopes, Relative Delta RSRP/SINR)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            PARALLEL INFERENCE (Next-Cell Classifier + Current Weakness Risk)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            PLATT SIGMOID CALIBRATION (Converts raw scores into genuine posterior probabilities)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            UNCERTAINTY &amp; SHANNON ENTROPY ESTIMATOR (Confidence = max P(k), H = -Σ p*log2(p))<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            ROUTE DIVERGENCE DETECTOR (Kinematic Bearing Vector vs Expected Corridors)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            DECOUPLED POLICY ENGINE (Cooldown Guardrails, Action Budgets, Quality Gates)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            EVENT LOGGING &amp; AUDIT STREAM (Structured Traceability)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ys-card">
        <div style="font-size: 13px; font-weight: 700; color: {current_theme['blue']}; margin-bottom: 8px;">
            2. CLAIM BOUNDARIES &amp; SAFETY CONSTRAINTS
        </div>
        <div style="font-size: 12px; color: {current_theme['text_secondary']}; line-height: 1.6;">
            <div>• <b>Simulation Proof-of-Concept</b>: Evaluated strictly on synthetic 3GPP-compliant propagation models and physical waypoint trajectories. The radio measurements are simulator outputs, not live operator traces.</div>
            <div>• <b>Advisory Shadow-Mode</b>: Outputs proactive candidate preparation recommendations to network controllers, avoiding destructive closed-loop interference.</div>
            <div>• <b>Honest Generalization</b>: Models require transfer learning fine-tuning when migrating across disparate carrier topologies and carrier frequencies.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ys-card">
        <div style="font-size: 13px; font-weight: 700; color: {current_theme['green']}; margin-bottom: 8px;">
            3. PHASED 5G &amp; O-RAN DEPLOYMENT ROADMAP
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 11.5px; color: {current_theme['text_secondary']};">
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['green']}; margin-bottom: 4px;">Phase 1 (Completed POC)</div>
                Calibrated XGBoost models, entropy uncertainty, route divergence engine, interactive console.
            </div>
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['accent']}; margin-bottom: 4px;">Phase 2 (Roadmap)</div>
                Kafka real-time telemetry streaming + high-throughput gRPC microservices.
            </div>
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['accent']}; margin-bottom: 4px;">Phase 3 (Roadmap)</div>
                Integration with Open5GS + OCUDU 5G Standalone (SA) software-defined radio lab.
            </div>
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['accent']}; margin-bottom: 4px;">Phase 4 (Roadmap)</div>
                Packaging as an <b>O-RAN Near-RT RIC xApp</b> over E2 KPM/RC service models.
            </div>
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['accent']}; margin-bottom: 4px;">Phase 5 (Roadmap)</div>
                Private 5G enterprise field trial in shadow evaluation mode.
            </div>
            <div style="background-color: {current_theme['bg_card_inner']}; padding: 12px; border-radius: 6px; border: 1px solid {current_theme['border']};">
                <div style="font-weight: 700; color: {current_theme['accent']}; margin-bottom: 4px;">Phase 6 (Roadmap)</div>
                Policy-guarded closed-loop mobility handover optimization.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 11. Subtle Enterprise Footer
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="ys-footer">
    <b>YELLOWSENSE</b> &nbsp;•&nbsp; Uncertainty-Aware Mobility Intelligence &nbsp;•&nbsp; Simulation-First Proof of Concept &nbsp;•&nbsp; COMET 5G Spring Board 6 | PS#6
</div>
""", unsafe_allow_html=True)
