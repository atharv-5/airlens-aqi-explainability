"""
LLM Conversational Layer for AQI Prediction.
Converts SHAP explainability values into plain-English policymaker insights.
Primary model: IBM Granite (via OpenRouter API).
Fallbacks: Google Gemini, Meta Llama, DeepSeek, and a robust rule-based engine.
"""

import os
import json
import urllib.request
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.utils import get_aqi_bucket

# Load environment variables
load_dotenv()


def format_shap_features_for_prompt(shap_features: List[Dict[str, Any]]) -> str:
    """Format top SHAP features into clear bullet points for the LLM prompt."""
    lines = []
    for item in shap_features:
        name = item.get("display_name", item.get("feature", "Feature"))
        val = item.get("feature_value", 0.0)
        shap_v = item.get("shap_value", 0.0)
        direction = item.get("direction", "influences AQI")
        lines.append(f"- {name} (Measured: {val:.1f}): attribution of {shap_v:+.1f} points ({direction})")
    return "\n".join(lines)


def get_rule_based_fallback(
    city: str,
    date: str,
    predicted_aqi: float,
    shap_features: List[Dict[str, Any]]
) -> str:
    """
    Intelligent rule-based explanation fallback in case LLM API key is missing or calls fail.
    Translates top SHAP drivers and CPCB health brackets into an actionable summary.
    """
    bucket_info = get_aqi_bucket(predicted_aqi)
    category = bucket_info["category"]
    advisory = bucket_info["advisory"]

    if not shap_features:
        return (
            f"The predicted AQI for {city} on {date} is {predicted_aqi:.0f}, which falls into the '{category}' category. "
            f"{advisory}"
        )

    # Identify primary upward driver (worsening AQI) and downward driver (improving AQI)
    upward_drivers = [f for f in shap_features if f.get("shap_value", 0) > 0]
    downward_drivers = [f for f in shap_features if f.get("shap_value", 0) < 0]

    up_feature = upward_drivers[0]["display_name"] if upward_drivers else None
    down_feature = downward_drivers[0]["display_name"] if downward_drivers else None

    parts = [
        f"On {date}, {city}'s air quality is projected at an AQI of {predicted_aqi:.0f} ('{category}'), posing {advisory.lower()}."
    ]

    if up_feature and down_feature:
        parts.append(
            f"The primary pressure pushing pollution levels higher is elevated {up_feature}, "
            f"partially mitigated by favorable reductions in {down_feature}."
        )
    elif up_feature:
        parts.append(
            f"This elevation is predominantly driven by heightened concentrations of {up_feature} across the metropolitan area."
        )
    elif down_feature:
        parts.append(
            f"Favorable meteorological factors and reduced {down_feature} help keep pollution levels subdued for this period."
        )

    if category in ("Very Poor", "Severe"):
        parts.append("Municipal authorities should consider strict vehicular restrictions and construction curbs.")
    elif category in ("Moderate", "Poor"):
        parts.append("Sensitive demographics, including children and asthmatics, are advised to minimize prolonged outdoor exertion.")
    else:
        parts.append("General conditions remain conducive to normal outdoor activities for all age cohorts.")

    return " ".join(parts)


def call_openrouter_api(prompt: str, api_key: str) -> str:
    """Invoke OpenRouter API with IBM Granite as primary model.
    
    Model priority:
    1. IBM Granite 4.2 8B (primary — IBM foundation model)
    2. IBM Granite 4.1 8B (IBM fallback)
    3. Google Gemini 2.5 Flash (secondary fallback)
    4. Meta Llama 3.3 70B (tertiary fallback)
    5. DeepSeek Chat (quaternary fallback)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AirLens AQI Explainability"
    }

    # IBM Granite models are prioritized, with other models as fallbacks
    models_to_try = [
        "ibm-granite/granite-4.2-8b",       # Primary: IBM Granite 4.2
        "google/gemini-2.5-flash",           # Fallback: Google Gemini
        "meta-llama/llama-3.3-70b-instruct", # Fallback: Meta Llama
        "deepseek/deepseek-chat",            # Fallback: DeepSeek
    ]

    for model_id in models_to_try:
        try:
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert environmental and air quality policy analyst. Provide short, compelling 2-3 sentence policy briefings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 600
            }

            # IBM Granite is a reasoning model — set low effort so it
            # doesn't consume the entire token budget on internal thinking
            if "granite" in model_id:
                payload["reasoning"] = {"effort": "low"}

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
                if "choices" in result and len(result["choices"]) > 0:
                    msg = result["choices"][0].get("message", {})
                    text = msg.get("content") or ""
                    text = text.strip() if text else ""
                    if text:
                        print(f"[llm_layer] [OK] Response from {model_id}")
                        return text
        except Exception as e:
            print(f"[llm_layer] OpenRouter attempt with {model_id} failed: {e}")
            continue

    raise RuntimeError("All OpenRouter models failed or key quota exceeded.")


def call_gemini_api(prompt: str, api_key: str) -> str:
    """Invoke direct Google Gemini SDK."""
    from google import genai
    client = genai.Client(api_key=api_key.strip())
    for model_id in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            continue
    raise RuntimeError("All Gemini SDK models failed.")


def generate_explanation(
    city: str,
    date: str,
    predicted_aqi: float,
    shap_features: List[Dict[str, Any]],
    api_key: str = None
) -> str:
    """
    Generate plain-English explanation for an AQI prediction and its SHAP factors.
    Auto-detects OpenRouter or Gemini API keys, with intelligent rule-based fallback.
    """
    shap_features_formatted = format_shap_features_for_prompt(shap_features)

    prompt = f"""You are an air quality analyst. Given this prediction, write a short, plain-English explanation (2-3 sentences) for a non-technical reader.

City: {city}
Date: {date}
Predicted AQI: {predicted_aqi:.1f}
Top contributing factors:
{shap_features_formatted}

Explain what is driving this AQI level and what it means for that day's air quality (e.g., "unhealthy for sensitive groups"). Do not repeat raw numbers robotically — write it like a short news-style insight."""

    # Detect API Key from parameter or environment
    key = (
        api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not key or key.strip() == "" or "your_" in key.lower():
        # Fallback when no key is configured
        return get_rule_based_fallback(city, date, predicted_aqi, shap_features)

    clean_key = key.strip()

    # Route to OpenRouter if key starts with sk-or- or OPENROUTER_API_KEY is present
    if clean_key.startswith("sk-or-") or os.getenv("OPENROUTER_API_KEY") == clean_key:
        try:
            print("[llm_layer] Using OpenRouter API...")
            return call_openrouter_api(prompt, clean_key)
        except Exception as e:
            print(f"[llm_layer] OpenRouter call failed: {e}. Using rule fallback.")
            return get_rule_based_fallback(city, date, predicted_aqi, shap_features)

    # Otherwise route to Gemini SDK
    try:
        print("[llm_layer] Using Google Gemini SDK...")
        return call_gemini_api(prompt, clean_key)
    except Exception as e:
        print(f"[llm_layer] Gemini API call failed: {e}. Using rule fallback.")
        return get_rule_based_fallback(city, date, predicted_aqi, shap_features)


if __name__ == "__main__":
    test_features = [
        {"feature": "PM2.5", "display_name": "Fine Particulate Matter (PM2.5)", "feature_value": 142.5, "shap_value": 48.3, "direction": "increases AQI"},
        {"feature": "aqi_rolling_3d", "display_name": "3-Day Prior AQI Trend", "feature_value": 210.0, "shap_value": 25.1, "direction": "increases AQI"},
        {"feature": "O3", "display_name": "Tropospheric Ozone (O3)", "feature_value": 18.0, "shap_value": -12.4, "direction": "decreases AQI"}
    ]
    explanation = generate_explanation("Delhi", "2026-09-21", 265.4, test_features)
    print("\n--- GENERATED EXPLANATION ---")
    print(explanation)
