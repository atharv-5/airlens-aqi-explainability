"""
Model Training module for AQI Prediction using XGBoost.
Implements time-based train/test splitting and evaluation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.data_prep import build_features, CLEAN_CSV_PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")


def time_based_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split time series data chronologically:
    First (1 - test_ratio) dates -> Train set
    Latest (test_ratio) dates -> Test set
    This strictly prevents lookahead data leakage.
    """
    df_sorted = df.sort_values(by="Date").reset_index(drop=True)
    unique_dates = df_sorted["Date"].drop_duplicates().sort_values().reset_index(drop=True)

    split_idx = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates.iloc[split_idx]

    train_df = df_sorted[df_sorted["Date"] < split_date].copy()
    test_df = df_sorted[df_sorted["Date"] >= split_date].copy()

    print(f"[train_model] Time-based split at date: {split_date}")
    print(f"[train_model] Train range: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} rows)")
    print(f"[train_model] Test range: {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} rows)")

    return train_df, test_df, split_date


def train_and_evaluate_model(clean_csv_path: str = CLEAN_CSV_PATH):
    """
    Train XGBRegressor on cleaned data, evaluate on time-held-out test set,
    and persist model and evaluation metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[train_model] Loading cleaned data from {clean_csv_path}...")
    df = pd.read_csv(clean_csv_path)

    # Perform time-based split
    train_df, test_df, split_date = time_based_train_test_split(df, test_ratio=0.2)

    # Build features for train and test
    X_train, y_train, feature_names = build_features(train_df)
    X_test, y_test, _ = build_features(test_df)

    # Align columns in case of missing categories in split
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[feature_names]

    print(f"[train_model] Training XGBRegressor with {len(feature_names)} features on {len(X_train)} rows...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("=" * 50)
    print("MODEL EVALUATION RESULTS (TIME-BASED HELD-OUT TEST SET)")
    print(f"  R² Score: {r2:.4f} (Benchmark: > 0.70)")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  MAE:      {mae:.2f}")
    print("=" * 50)

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print(f"[train_model] Saved trained XGBoost model to {MODEL_PATH}")

    # Save feature names
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metrics metadata
    metrics_data = {
        "model_type": "XGBoost Regressor (Tree-based ensemble)",
        "split_type": "Time-based chronological split (80/20)",
        "split_date": str(split_date),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "features": feature_names,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[train_model] Saved metrics to {METRICS_PATH}")

    return model, X_test, y_test, feature_names, metrics_data


if __name__ == "__main__":
    train_and_evaluate_model()
