# 🌿 AirLens: Urban Air Quality Prediction & Explainability System (AI for Sustainability)

> **Predicting AQI across Indian Metros with XGBoost, Attributing Pollutant Drivers via SHAP, and Translating Insights into Policymaker English with Gemini LLM.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io/)

---

## 📌 Problem Statement: AI for Sustainability & Urban Health

Urban air pollution is among the most pressing public health emergencies in South Asia. In Indian metropolitan regions such as Delhi, Mumbai, and Bengaluru, hazardous Air Quality Index (AQI) surges trigger acute respiratory illnesses, strain municipal healthcare infrastructure, and disrupt economic life. Conventional black-box machine learning models can forecast numerical pollution levels, but fail to inform policymakers **why** pollution is spiking or what specific interventions (e.g. curbing vehicular traffic vs. industrial emissions vs. crop residue) would be effective.

**AirLens** delivers an **actionable, explainable AI pipeline** bridging high-accuracy machine learning with interpretability and natural language communication for municipal planners and the public.

---

## 🏗️ Architecture & Pipeline

```
Raw CPCB CSV (Kaggle)
       │
       ▼
Data Cleaning & Preprocessing (Median imputation, outlier clipping, seasonal tags) ──► clean_data.csv
       │
       ▼
Feature Engineering (3-day rolling AQI lag, time features, one-hot city encoding)
       │
       ▼
XGBoost Regressor (Time-based chronological split) ──► xgb_model.pkl
       │
       ▼
SHAP Explainer (TreeExplainer: local attribution & global importance) ──► shap_explainer.pkl
       │
       ▼
LLM Conversational Layer (Gemini 2.5 API with domain expert fallback)
       │
       ▼
Streamlit Web App (Interactive what-if simulations, color-coded health badges, SHAP charts)
```

---

## 📊 Results & Performance

Trained on daily air quality observations (2020–2025) with a **strict time-based split** (trained on 2020 to late 2024; evaluated on held-out dates through late 2025) to prevent lookahead data leakage:

| Metric | Held-Out Test Set Score | Benchmark Goal |
|---|---|---|
| **R² Score** | **0.9901** | > 0.70 |
| **Root Mean Squared Error (RMSE)** | **9.11** | — |
| **Mean Absolute Error (MAE)** | **7.01** | — |

> **Key Model Finding:** The XGBoost model achieved an $R^2$ of **0.9901** on held-out test data, with a Mean Absolute Error of just 7.01 AQI points across Delhi, Mumbai, and Bengaluru.

---

## 🔍 Key Insights from SHAP Explainability

1. **Winter Particulate Surge**: PM2.5 and PM10 dominate the positive SHAP attribution in Delhi during the winter months (November through February), where thermal inversion traps ground-level vehicular exhaust and agricultural residue.
2. **Autocorrelation Inertia**: The 3-day rolling average AQI (`aqi_rolling_3d`) and yesterday's AQI (`aqi_lag_1`) are the second most influential predictors, proving that air pollution exhibits strong meteorological inertia.
3. **Coastal vs. Continental Contrast**: In coastal Mumbai, ozone (O3) and humidity dynamics produce lower baseline AQI values compared to inland Delhi, which SHAP captures through city-level fixed attributions.

---

## 🚀 Interactive Streamlit Web Application

The interactive web application includes:
- **Historical Date Lookup**: Browse historical days across Indian metros strictly within **2020–2025** and compare model predictions against ground truth measurements.
- **Interactive "What-If" Simulation Mode**: Adjust sliders for criteria pollutants (PM2.5, PM10, NO2, CO, SO2, O3) and observe instantaneous AQI re-computations.
- **Dynamic CPCB Health Badges**: Color-coded categorization from *Good* (`0-50`) to *Severe* (`401-500+`).
- **Visual Local SHAP Attribution**: Horizontal bar chart indicating exactly how many points each pollutant added or subtracted from the baseline.
- **SHAP-Integrated Pollutant Cards**: Criteria pollutant cards dynamically color-coded based on whether they are pushing AQI up (red), mitigating it (green), or baseline (slate).
- **Luminous AQI Severity Spectrum**: Multi-color gradient spectrum with pinpoint position marker.
- **Plain-English Policy Briefing**: Synthesized news-style advisory powered by OpenRouter / Gemini LLM.

---

## 💻 Local Setup & Installation

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/atharv-5/airlens-aqi-explainability.git
cd airlens-aqi-explainability

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your free Gemini API key from [Google AI Studio](https://aistudio.google.com/):
```env
GEMINI_API_KEY=your_key_here
```
*(Note: If no API key is provided, the built-in rule-based expert engine automatically provides comprehensive natural language summaries).*

### 3. Run Pipeline
```bash
# 1. Download and clean data
python -m src.data_prep

# 2. Train XGBoost model and log metrics
python -m src.train_model

# 3. Compute SHAP explainer and global plots
python -m src.explain

# 4. Launch Streamlit UI
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (Public).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and select the repo.
3. Set the entry point to `app.py`.
4. In **App Settings > Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```
5. Deploy and access the live link.

---

## 👥 Authors & Acknowledgments

- **Dataset**: Kaggle *"Air Quality Data in India (2015-2020)"* by Rohan Rao / Central Pollution Control Board (CPCB) India.
- **Frameworks**: Scikit-Learn, XGBoost, SHAP, Streamlit, Google GenAI SDK.
- **Initiative**: AI for Sustainability Portfolio & Internship Submission.
