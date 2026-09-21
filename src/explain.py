"""
SHAP Explainability module for AQI prediction.
Uses TreeExplainer to generate local feature attributions and global feature importance.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import shap

from src.data_prep import build_features, CLEAN_CSV_PATH
from src.train_model import MODEL_PATH, MODELS_DIR
from src.utils import format_feature_name

EXPLAINER_PATH = os.path.join(MODELS_DIR, "shap_explainer.pkl")
SUMMARY_PLOT_PATH = os.path.join(MODELS_DIR, "shap_summary.png")


def get_shap_explainer(model, X_background: pd.DataFrame = None) -> shap.TreeExplainer:
    """
    Initialize a shap.TreeExplainer for the trained XGBoost model.
    Caches or loads the explainer from disk if available.
    """
    if os.path.exists(EXPLAINER_PATH):
        try:
            print(f"[explain] Loading cached SHAP explainer from {EXPLAINER_PATH}...")
            explainer = joblib.load(EXPLAINER_PATH)
            return explainer
        except Exception as e:
            print(f"[explain] Warning: could not load cached explainer ({e}), recomputing...")

    print("[explain] Creating new shap.TreeExplainer for XGBoost model...")
    # TreeExplainer is native and fast for tree models like XGBoost
    explainer = shap.TreeExplainer(model)

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(explainer, EXPLAINER_PATH)
        print(f"[explain] Saved SHAP explainer to {EXPLAINER_PATH}")
    except Exception as e:
        print(f"[explain] Notice: explainer dump failed ({e}), continuing in-memory.")

    return explainer


def explain_prediction(explainer: shap.TreeExplainer, X_row: pd.DataFrame, top_k: int = 5) -> dict:
    """
    Explain a single prediction row:
    Returns the top K features by absolute SHAP value, indicating their value,
    SHAP attribution score, direction (increases or decreases AQI), and readable labels.
    """
    if isinstance(X_row, pd.Series):
        X_row = X_row.to_frame().T

    # Compute SHAP values for the single sample
    shap_vals = explainer(X_row)
    values = shap_vals.values[0]
    base_val = float(shap_vals.base_values[0]) if hasattr(shap_vals.base_values, "__len__") else float(shap_vals.base_values)
    feature_names = list(X_row.columns)

    # Compile feature impact records
    features_data = []
    for f_name, raw_val, s_val in zip(feature_names, X_row.iloc[0], values):
        features_data.append({
            "feature": f_name,
            "display_name": format_feature_name(f_name),
            "feature_value": float(raw_val),
            "shap_value": float(s_val),
            "abs_shap": abs(float(s_val)),
            "direction": "increases AQI" if s_val > 0 else "decreases AQI",
            "impact": "Worsens air quality (+)" if s_val > 0 else "Improves air quality (-)"
        })

    # Sort by absolute SHAP attribution
    features_data.sort(key=lambda x: x["abs_shap"], reverse=True)
    top_features = features_data[:top_k]

    return {
        "base_value": round(base_val, 2),
        "top_features": top_features,
        "all_shap_values": values,
        "feature_names": feature_names,
        "shap_explanation_obj": shap_vals
    }


def generate_summary_plot(explainer: shap.TreeExplainer, X_sample: pd.DataFrame, output_path: str = SUMMARY_PLOT_PATH):
    """
    Generate and save a global SHAP summary bar plot for feature importance.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[explain] Computing SHAP values for {len(X_sample)} sample rows to generate summary plot...")
    
    # Subsample if large for speed
    if len(X_sample) > 500:
        X_sample = X_sample.sample(500, random_state=42)

    shap_values = explainer(X_sample)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Create clean horizontal bar plot of mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    
    y_pos = np.arange(len(top_indices))
    labels = [format_feature_name(X_sample.columns[i]) for i in top_indices]
    scores = mean_abs_shap[top_indices]

    colors = ["#d9534f" if i == 0 else "#2b5c8f" for i in range(len(top_indices))]
    bars = ax.barh(y_pos, scores[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=10, fontweight="medium")
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on AQI Prediction)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Global Feature Importance (TreeExplainer SHAP)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add numeric labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max(scores) * 0.01), bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                va="center", ha="left", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[explain] Saved global SHAP summary plot to {output_path}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        from src.train_model import train_and_evaluate_model
        model, _, _, _, _ = train_and_evaluate_model()
    else:
        model = joblib.load(MODEL_PATH)

    df = pd.read_csv(CLEAN_CSV_PATH)
    X, y, feature_names = build_features(df)

    explainer = get_shap_explainer(model)
    generate_summary_plot(explainer, X)

    # Test local prediction explanation on first row
    sample_row = X.iloc[[0]]
    explanation = explain_prediction(explainer, sample_row)
    print("\n--- SAMPLE LOCAL SHAP EXPLANATION (Row 0) ---")
    print(f"Base AQI: {explanation['base_value']}")
    for f in explanation["top_features"]:
        print(f"  * {f['display_name']}: {f['feature_value']:.2f} -> SHAP {f['shap_value']:+.2f} ({f['direction']})")
