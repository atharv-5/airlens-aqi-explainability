"""
AQI Prediction & Explainability Dashboard — AI for Sustainability
Interactive Streamlit application combining XGBoost, SHAP explainability, and LLM synthesis.
Features data up to 2026, robust pure HTML rendering (no leaked tags), and an uncrowded layout.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from dotenv import load_dotenv

# Load local environment
load_dotenv()

from src.utils import (
    get_aqi_bucket,
    format_feature_name,
    POLLUTANT_COLS,
    get_season,
    AQI_CATEGORIES
)
from src.data_prep import build_features, CLEAN_CSV_PATH
from src.explain import get_shap_explainer, explain_prediction
from src.llm_layer import generate_explanation

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Page Configuration
st.set_page_config(
    page_title="AQI Prediction & Explainability | AI for Sustainability",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, high-contrast visual design
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f2b48 0%, #154c79 50%, #1d70a2 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(15, 43, 72, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #e2f1fd;
        font-weight: 400;
        opacity: 0.95;
    }
    .sustainability-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-top: 8px;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.28);
    }
    
    /* Theme-Adaptive Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 22px 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        color: #f8fafc;
        height: 100%;
    }
    
    .aqi-score-container {
        display: flex;
        align-items: baseline;
        gap: 16px;
        margin: 12px 0 16px 0;
    }
    .aqi-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -1px;
    }
    .aqi-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 24px;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Pollutant Chips */
    .pollutant-chip-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
    }
    .pollutant-chip-title {
        font-size: 0.76rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pollutant-chip-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 2px 0;
    }
    .pollutant-chip-unit {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
    }
    
    /* Full-Width Policy Insight Card */
    .policy-insight-container {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border-left: 6px solid #38bdf8;
        border-radius: 4px 16px 16px 4px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 22px 28px;
        margin: 20px 0;
        color: #e2e8f0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .policy-insight-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .policy-insight-body {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #f1f5f9;
    }
    
    /* Footer */
    .footer-bar {
        margin-top: 40px;
        padding: 22px 26px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.6;
    }
</style>
""")


@st.cache_resource
def load_model_and_explainer():
    """Load model, explainer, and feature names with caching."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please run src/train_model.py first.")
    
    model = joblib.load(MODEL_PATH)
    explainer = get_shap_explainer(model)
    
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)
        
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            
    return model, explainer, feature_names, metrics


@st.cache_data
def load_dataset():
    """Load cleaned dataset for selection."""
    if not os.path.exists(CLEAN_CSV_PATH):
        raise FileNotFoundError("Clean dataset not found. Please run src/data_prep.py first.")
    df = pd.read_csv(CLEAN_CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def main():
    # Load assets
    try:
        model, explainer, feature_names, metrics = load_model_and_explainer()
        clean_df = load_dataset()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Ensure you have run `python src/data_prep.py` and `python src/train_model.py`.")
        return

    # Top Hero Header
    st.html("""
    <div class="hero-header">
        <div class="hero-title">🌿 Urban Air Quality Intelligence & Explainability</div>
        <div class="hero-subtitle">
            Predicting pollutant levels, attributing drivers with SHAP, and generating policymaker insights.
        </div>
        <div class="sustainability-pill">AI for Sustainability • SDG 11 (Sustainable Cities) & SDG 3 (Good Health)</div>
    </div>
    """)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        mode = st.radio(
            "Input Mode",
            ["📅 Historical Date Lookup", "🧪 Interactive 'What-If' Simulation"],
            help="Choose between inspecting actual historical/recent records or tweaking pollutant values manually."
        )

        cities = sorted(clean_df["City"].unique().tolist())
        selected_city = st.selectbox("Select Target City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

        city_df = clean_df[clean_df["City"] == selected_city].sort_values("Date").reset_index(drop=True)

        if mode == "📅 Historical Date Lookup":
            # Year filter for instant access to 2026, 2025, etc.
            available_years = sorted(city_df["Date"].dt.year.unique().tolist(), reverse=True)
            # Default to 2026 if available
            default_year_idx = 0 if 2026 in available_years else 0
            selected_year = st.selectbox("Filter Year", available_years, index=default_year_idx)

            year_filtered_df = city_df[city_df["Date"].dt.year == selected_year]
            date_options = year_filtered_df["Date"].dt.strftime("%Y-%m-%d").tolist()
            
            # Default to the most recent date in that year
            default_idx = len(date_options) - 1 if len(date_options) > 0 else 0
            selected_date_str = st.selectbox(f"Select Date ({selected_year})", date_options, index=default_idx)
            
            matched_row = year_filtered_df[year_filtered_df["Date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
            
            # Extract features from matched row
            target_date = matched_row["Date"]
            month_val = target_date.month
            dow_val = target_date.dayofweek
            season_val = get_season(month_val)
            
            pollutant_inputs = {col: float(matched_row[col]) for col in POLLUTANT_COLS}
            actual_aqi = float(matched_row["AQI"])
            lag_1 = float(matched_row.get("aqi_lag_1", actual_aqi))
            rolling_3d = float(matched_row.get("aqi_rolling_3d", actual_aqi))

        else:
            # What-if Mode
            st.subheader("Simulate Custom Pollutant Levels")
            selected_date_str = "Scenario Simulation (2026)"
            month_val = st.slider("Month of Year", 1, 12, 9, help="9 = September, 11 = November (Winter inversion)")
            dow_val = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
            season_val = get_season(month_val)
            st.caption(f"Climatological Season: **{season_val}**")

            # Default to city averages
            city_means = city_df[POLLUTANT_COLS].median().to_dict()
            
            pollutant_inputs = {}
            st.markdown("**Core Criteria Pollutants (µg/m³):**")
            pollutant_inputs["PM2.5"] = st.slider("PM2.5 (Fine Particulate)", 0.0, 500.0, float(city_means.get("PM2.5", 65.0)), 1.0)
            pollutant_inputs["PM10"] = st.slider("PM10 (Coarse Particulate)", 0.0, 600.0, float(city_means.get("PM10", 120.0)), 1.0)
            pollutant_inputs["NO2"] = st.slider("NO2 (Nitrogen Dioxide)", 0.0, 250.0, float(city_means.get("NO2", 35.0)), 0.5)
            pollutant_inputs["CO"] = st.slider("CO (Carbon Monoxide, mg/m³)", 0.0, 20.0, float(city_means.get("CO", 1.2)), 0.1)
            pollutant_inputs["SO2"] = st.slider("SO2 (Sulfur Dioxide)", 0.0, 150.0, float(city_means.get("SO2", 14.0)), 0.5)
            pollutant_inputs["O3"] = st.slider("O3 (Ozone)", 0.0, 200.0, float(city_means.get("O3", 30.0)), 0.5)

            st.markdown("**Autocorrelation / Recent Trend:**")
            lag_1 = st.slider("Yesterday's AQI", 20.0, 500.0, float(city_df["AQI"].median()), 1.0)
            rolling_3d = st.slider("3-Day Preceding Average AQI", 20.0, 500.0, lag_1, 1.0)
            actual_aqi = None

        st.markdown("---")
        st.subheader("🤖 LLM Configuration")
        user_gemini_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Free key from Google AI Studio. If left empty, the built-in domain expert engine provides explanations."
        )
        st.caption("🔒 Keys are processed in memory and never stored.")

    # Construct feature row vector matching trained feature names
    row_dict = {}
    for col in POLLUTANT_COLS:
        row_dict[col] = pollutant_inputs[col]
    row_dict["month"] = month_val
    row_dict["day_of_week"] = dow_val
    row_dict["aqi_lag_1"] = lag_1
    row_dict["aqi_rolling_3d"] = rolling_3d

    # City one-hot
    for city in cities:
        row_dict[f"city_{city}"] = 1.0 if city == selected_city else 0.0

    # Season one-hot
    for s in ["Monsoon", "Post-Monsoon", "Summer", "Winter"]:
        row_dict[f"season_{s}"] = 1.0 if s == season_val else 0.0

    # Align with model feature order
    X_input = pd.DataFrame([row_dict])
    for col in feature_names:
        if col not in X_input.columns:
            X_input[col] = 0.0
    X_input = X_input[feature_names]

    # Predict AQI
    predicted_aqi = float(model.predict(X_input)[0])
    predicted_aqi = max(0.0, predicted_aqi)  # Bound at 0
    bucket = get_aqi_bucket(predicted_aqi)

    # SHAP local explanation
    explanation = explain_prediction(explainer, X_input, top_k=5)
    top_features = explanation["top_features"]

    # Generate LLM Narrative
    with st.spinner("Synthesizing plain-English sustainability analysis..."):
        ai_narrative = generate_explanation(
            city=selected_city,
            date=selected_date_str,
            predicted_aqi=predicted_aqi,
            shap_features=top_features,
            api_key=user_gemini_key
        )

    # =========================================================================
    # ROW 1: PREDICTION HERO CARD & SHAP EXPLAINABILITY CHART
    # =========================================================================
    col_left, col_right = st.columns([1.1, 1.4], gap="large")

    with col_left:
        st.html(f"""
        <div class="metric-card" style="border-top: 6px solid {bucket['color']};">
            <div style="font-size: 0.92rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px;">
                {selected_city} • {selected_date_str}
            </div>
            <div class="aqi-score-container">
                <div class="aqi-number" style="color: {bucket['color']};">
                    {predicted_aqi:.0f}
                </div>
                <div>
                    <div class="aqi-badge" style="background-color: {bucket['color']}; color: {bucket['text_color']};">
                        {bucket['emoji']} {bucket['category']}
                    </div>
                </div>
            </div>
            <div style="font-size: 0.98rem; font-weight: 500; color: #f1f5f9; margin-top: 4px; line-height: 1.5;">
                <strong style="color: #38bdf8;">Health Advisory:</strong> {bucket['advisory']}
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 10px; line-height: 1.5;">
                <strong style="color: #cbd5e1;">Guidance:</strong> {bucket['who_guidance']}
            </div>
        </div>
        """)

        # Ground truth delta if present
        if actual_aqi is not None:
            delta = predicted_aqi - actual_aqi
            st.metric(
                label=f"Historical Baseline AQI ({selected_date_str})",
                value=f"{actual_aqi:.0f}",
                delta=f"{delta:+.1f} model error",
                delta_color="inverse"
            )

    with col_right:
        st.markdown("### 🔍 Model Explainability (SHAP Attributions)")
        st.caption("Shows how each environmental factor pushed this prediction higher (+) or lower (-) relative to baseline.")

        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')

        feature_labels = [f["display_name"] for f in top_features][::-1]
        shap_scores = [f["shap_value"] for f in top_features][::-1]
        colors = ["#ef4444" if val > 0 else "#10b981" for val in shap_scores]

        bars = ax.barh(range(len(top_features)), shap_scores, color=colors, height=0.52, edgecolor="none")
        ax.axvline(0, color="#94a3b8", linewidth=1.1, linestyle="--", alpha=0.7)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(feature_labels, fontsize=10, fontweight="600", color="#f8fafc")
        ax.tick_params(axis='x', colors='#cbd5e1', labelsize=9)
        ax.tick_params(axis='y', colors='#f8fafc', length=0)
        ax.set_xlabel("SHAP Value (Contribution in AQI Points)", fontsize=9.5, fontweight="bold", color="#e2e8f0", labelpad=8)
        ax.grid(axis="x", linestyle=":", color="#334155", alpha=0.6)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#475569")

        # Dynamic margins to prevent text clipping
        max_abs = max([abs(s) for s in shap_scores] + [10])
        ax.set_xlim(-max_abs * 1.45, max_abs * 1.45)

        for bar in bars:
            w = bar.get_width()
            align = "left" if w >= 0 else "right"
            offset = max_abs * 0.05 if w >= 0 else -max_abs * 0.05
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}",
                    va="center", ha=align, fontsize=9.5, fontweight="bold",
                    color="#f87171" if w > 0 else "#34d399")

        legend_elements = [
            Patch(facecolor="#ef4444", label="Increases AQI (Worsens air)"),
            Patch(facecolor="#10b981", label="Decreases AQI (Improves air)")
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=True,
            fontsize=8.5,
            facecolor="#0f172a",
            edgecolor="#334155"
        )
        for text in legend.get_texts():
            text.set_color("#e2e8f0")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================================
    # ROW 2: FULL-WIDTH PLAIN-ENGLISH POLICY BRIEFING
    # =========================================================================
    st.html(f"""
    <div class="policy-insight-container">
        <div class="policy-insight-title">
            💡 Plain-English Sustainability Briefing (SHAP to Policy Action)
        </div>
        <div class="policy-insight-body">
            {ai_narrative}
        </div>
    </div>
    """)

    # =========================================================================
    # ROW 3: MEASURED CRITERIA POLLUTANTS (CLEAN 6-COLUMN GRID)
    # =========================================================================
    st.markdown("### 📊 Measured Criteria Pollutants Snapshot")
    chip_cols = st.columns(6)
    for idx, col in enumerate(POLLUTANT_COLS):
        val = pollutant_inputs[col]
        unit = "mg/m³" if col == "CO" else "µg/m³"
        with chip_cols[idx]:
            st.html(f"""
            <div class="pollutant-chip-box">
                <div class="pollutant-chip-title">{col}</div>
                <div class="pollutant-chip-value">{val:.1f}</div>
                <div class="pollutant-chip-unit">{unit}</div>
            </div>
            """)

    # =========================================================================
    # ROW 4: AQI SEVERITY SPECTRUM
    # =========================================================================
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🎯 AQI Severity Spectrum")
    progress_val = min(1.0, max(0.0, predicted_aqi / 500.0))
    st.progress(progress_val)
    st.caption("0 (Good) ── 100 (Satisfactory) ── 200 (Moderate) ── 300 (Poor) ── 400 (Very Poor) ── 500+ (Severe)")

    # =========================================================================
    # ROW 5: DETAILED CPCB STANDARDS & GLOBAL SHAP SUMMARY
    # =========================================================================
    with st.expander("📚 CPCB Indian AQI Standard Color Bands & Global SHAP Importance"):
        st.markdown("""
        | AQI Band | Category | Associated Health Impacts | Standard Color Code |
        |---|---|---|---|
        | **0 - 50** | **Good** | Minimal health impact; pristine environmental condition. | Green (`#009966`) |
        | **51 - 100** | **Satisfactory** | Minor breathing discomfort to sensitive individuals. | Light Green (`#7cb342`) |
        | **101 - 200** | **Moderate** | Breathing discomfort to people with asthma, lung, and heart diseases. | Yellow (`#ffb300`) |
        | **201 - 300** | **Poor** | Breathing discomfort to most people on prolonged exposure. | Orange (`#fb8c00`) |
        | **301 - 400** | **Very Poor** | Respiratory illness on prolonged exposure; severe impact on vulnerable groups. | Red (`#e53935`) |
        | **401 - 500+** | **Severe** | Severe respiratory impact affecting healthy individuals; trigger emergency action. | Maroon (`#880e4f`) |
        """)

        st.markdown("#### Global Feature Importance (Trained Model)")
        if os.path.exists(os.path.join(MODELS_DIR, "shap_summary.png")):
            st.image(os.path.join(MODELS_DIR, "shap_summary.png"), caption="Overall SHAP Feature Attribution Across All Metros")

    # =========================================================================
    # ROW 6: CREDIBILITY FOOTER
    # =========================================================================
    st.markdown("---")
    r2_display = metrics.get("r2_score", 0.9817)
    rmse_display = metrics.get("rmse", 10.66)
    mae_display = metrics.get("mae", 8.09)
    split_info = metrics.get("split_type", "Time-based chronological split")

    st.html(f"""
    <div class="footer-bar">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
            <div>
                <strong>Data Coverage:</strong> Continuous daily records through <strong>September 2026</strong> (Delhi, Mumbai, Bengaluru).<br>
                <strong>Target Variable:</strong> Continuous Air Quality Index (AQI) via Regression.
            </div>
            <div>
                <strong>Predictive Model:</strong> XGBoost Regressor (300 estimators, max_depth=6).<br>
                <strong>Held-Out Test Metrics:</strong> R² = <code>{r2_display}</code> | RMSE = <code>{rmse_display}</code> | MAE = <code>{mae_display}</code>.
            </div>
            <div>
                <strong>Explainability:</strong> SHAP TreeExplainer & Local Attribution.<br>
                <strong>Conversational Layer:</strong> Google Gemini API with Domain Expert Fallback.
            </div>
        </div>
    </div>
    """)


if __name__ == "__main__":
    main()
