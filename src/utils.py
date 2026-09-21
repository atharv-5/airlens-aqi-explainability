"""
Utility functions and configuration for AQI Prediction and Explainability system.
Adheres to Central Pollution Control Board (CPCB) India AQI standards.
"""

from typing import Dict, Any

# AQI Categories and standard CPCB color coding
AQI_CATEGORIES = [
    {
        "min": 0,
        "max": 50,
        "category": "Good",
        "color": "#009966",
        "light_color": "#E6F4EA",
        "badge_color": "#28a745",
        "text_color": "#FFFFFF",
        "emoji": "🌱",
        "advisory": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "who_guidance": "Enjoy outdoor activities. Minimal impact on health."
    },
    {
        "min": 51,
        "max": 100,
        "category": "Satisfactory",
        "color": "#7cb342",
        "light_color": "#F1F8E9",
        "badge_color": "#8bc34a",
        "text_color": "#FFFFFF",
        "emoji": "🌤️",
        "advisory": "Minor breathing discomfort to sensitive people.",
        "who_guidance": "Acceptable air quality; unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    },
    {
        "min": 101,
        "max": 200,
        "category": "Moderate",
        "color": "#ffb300",
        "light_color": "#FFF8E1",
        "badge_color": "#ffc107",
        "text_color": "#212529",
        "emoji": "⛅",
        "advisory": "Breathing discomfort to the people with lungs, asthma and heart diseases.",
        "who_guidance": "Sensitive groups should reduce heavy outdoor exertion; wear a mask if experiencing respiratory irritation."
    },
    {
        "min": 201,
        "max": 300,
        "category": "Poor",
        "color": "#fb8c00",
        "light_color": "#FFF3E0",
        "badge_color": "#fd7e14",
        "text_color": "#FFFFFF",
        "emoji": "😷",
        "advisory": "Breathing discomfort to most people on prolonged exposure.",
        "who_guidance": "Limit prolonged outdoor exertion. Children and elderly should remain indoors."
    },
    {
        "min": 301,
        "max": 400,
        "category": "Very Poor",
        "color": "#e53935",
        "light_color": "#FFEBEE",
        "badge_color": "#dc3545",
        "text_color": "#FFFFFF",
        "emoji": "⚠️",
        "advisory": "Respiratory illness on prolonged exposure. Pronounced effect on people with lung and heart diseases.",
        "who_guidance": "Avoid outdoor activities; keep indoor air purifiers running and seal drafty windows."
    },
    {
        "min": 401,
        "max": float("inf"),
        "category": "Severe",
        "color": "#880e4f",
        "light_color": "#FCE4EC",
        "badge_color": "#6a1b9a",
        "text_color": "#FFFFFF",
        "emoji": "🚨",
        "advisory": "Affects healthy people and seriously impacts those with existing diseases.",
        "who_guidance": "Public health emergency level. Emergency health warnings triggered. Remain strictly indoors."
    },
]

# Core pollutants in Kaggle dataset
POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
ALL_POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

# Standard realistic sensor clipping limits (ug/m3, CO in mg/m3)
POLLUTANT_BOUNDS = {
    "PM2.5": (0.0, 1000.0),
    "PM10": (0.0, 1500.0),
    "NO": (0.0, 500.0),
    "NO2": (0.0, 500.0),
    "NOx": (0.0, 600.0),
    "NH3": (0.0, 500.0),
    "CO": (0.0, 50.0),
    "SO2": (0.0, 500.0),
    "O3": (0.0, 500.0),
    "Benzene": (0.0, 300.0),
    "Toluene": (0.0, 300.0),
    "Xylene": (0.0, 300.0),
    "AQI": (0.0, 1000.0),
}

def get_aqi_bucket(aqi_value: float) -> Dict[str, Any]:
    """Return the AQI category, color metadata, and health advice for a given AQI."""
    if aqi_value is None or aqi_value < 0:
        return {
            "category": "Unknown",
            "color": "#6c757d",
            "light_color": "#f8f9fa",
            "badge_color": "#6c757d",
            "text_color": "#FFFFFF",
            "emoji": "❓",
            "advisory": "Data unavailable or invalid value.",
            "who_guidance": "No advisory available."
        }
    for item in AQI_CATEGORIES:
        if item["min"] <= aqi_value <= item["max"]:
            return item
    return AQI_CATEGORIES[-1]


def get_season(month: int) -> str:
    """Map month (1-12) to Indian climatological season."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:
        return "Post-Monsoon"


def format_feature_name(feature_name: str) -> str:
    """Make technical feature names polished and human-readable."""
    mapping = {
        "PM2.5": "Fine Particulate Matter (PM2.5)",
        "PM10": "Coarse Particulate Matter (PM10)",
        "NO2": "Nitrogen Dioxide (NO2)",
        "SO2": "Sulfur Dioxide (SO2)",
        "CO": "Carbon Monoxide (CO)",
        "O3": "Tropospheric Ozone (O3)",
        "month": "Month of Year",
        "day_of_week": "Day of Week",
        "aqi_rolling_3d": "3-Day Prior AQI Trend",
        "aqi_lag_1": "Yesterday's AQI",
    }
    if feature_name in mapping:
        return mapping[feature_name]
    if feature_name.startswith("city_"):
        city = feature_name.replace("city_", "").replace("_", " ").title()
        return f"Location: {city}"
    if feature_name.startswith("season_"):
        season = feature_name.replace("season_", "").title()
        return f"Season: {season}"
    return feature_name.replace("_", " ").title()
