"""
Data Preparation and Feature Engineering module for AQI Prediction.
Processes Kaggle "Air Quality Data in India" (city_day.csv).
"""

import os
import urllib.request
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from src.utils import POLLUTANT_COLS, ALL_POLLUTANT_COLS, POLLUTANT_BOUNDS, get_season

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_CSV_PATH = os.path.join(DATA_RAW_DIR, "city_day.csv")
CLEAN_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "clean_data.csv")

# Reliable mirror of Kaggle "Air Quality Data in India" city_day.csv
DATASET_URL = (
    "https://raw.githubusercontent.com/HarshiSharma04/AQI_Predictor_using_Time_Series_Analysis/master/city_day.csv"
)

# Core focus cities recommended in the roadmap
DEFAULT_CITIES = ["Delhi", "Bengaluru", "Mumbai"]


def download_dataset_if_needed(target_path: str = RAW_CSV_PATH) -> str:
    """Ensure the raw Kaggle dataset is present, downloading if necessary."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000:
        print(f"[data_prep] Downloading Kaggle Air Quality dataset to {target_path}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, target_path)
            print(f"[data_prep] Download completed ({os.path.getsize(target_path) / 1024:.1f} KB).")
        except Exception as e:
            raise RuntimeError(
                f"Failed to automatically download dataset from {DATASET_URL}: {e}. "
                f"Please place city_day.csv manually in {DATA_RAW_DIR}."
            )
    else:
        print(f"[data_prep] Raw dataset already exists at {target_path}.")
    return target_path


def load_raw_data(filepath: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load raw city_day.csv from disk."""
    if not os.path.exists(filepath):
        download_dataset_if_needed(filepath)
    df = pd.read_csv(filepath)
    return df


def clean_air_quality_data(
    df: pd.DataFrame,
    target_cities: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean and preprocess raw air quality data:
    1. Filter to selected target cities.
    2. Drop rows where target AQI is missing.
    3. Impute pollutant missing values using city-wise median and forward fill.
    4. Clip sensor outliers to physically plausible limits.
    5. Parse Date and extract temporal features (year, month, day_of_week, season).
    6. Calculate lag & rolling average features per city.
    """
    if target_cities is None:
        target_cities = DEFAULT_CITIES

    print(f"[data_prep] Filtering for cities: {target_cities}...")
    df = df[df["City"].isin(target_cities)].copy()

    # Drop records missing target AQI
    initial_len = len(df)
    df = df.dropna(subset=["AQI"]).copy()
    print(f"[data_prep] Dropped {initial_len - len(df)} rows missing AQI target. Remaining: {len(df)}")

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)

    # Clip outliers for all available pollutant columns and AQI
    for col, (lower_b, upper_b) in POLLUTANT_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower_b, upper=upper_b)

    # Impute missing values for pollutants:
    # 1. Forward-fill along time within each city
    # 2. Backward-fill along time within each city
    # 3. City-wise median imputation for any remaining NaNs
    for city in target_cities:
        city_mask = df["City"] == city
        city_sub = df.loc[city_mask]

        for col in ALL_POLLUTANT_COLS:
            if col in df.columns:
                # Time series forward & backward fill per city
                df.loc[city_mask, col] = city_sub[col].ffill().bfill()
                # If still NaN (e.g. Entire column was empty for that city), fill with city median or column median
                city_med = df.loc[city_mask, col].median()
                if pd.isna(city_med):
                    city_med = df[col].median()
                if pd.notna(city_med):
                    df.loc[city_mask, col] = df.loc[city_mask, col].fillna(city_med)

    # Drop any remaining unfillable pollutant NaNs for the core pollutants
    df = df.dropna(subset=POLLUTANT_COLS).copy()

    # Date feature engineering
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["season"] = df["month"].apply(get_season)

    # Time-series autocorrelation features per city:
    # 1. Yesterday's AQI (lag_1)
    # 2. 3-day rolling average of AQI (lagged so it represents the preceding 3 days, not future)
    df["aqi_lag_1"] = df.groupby("City")["AQI"].shift(1)
    df["aqi_rolling_3d"] = (
        df.groupby("City")["AQI"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # For the first rows where lag is NaN, backfill with the first observed AQI in that city
    df["aqi_lag_1"] = df.groupby("City")["aqi_lag_1"].bfill()
    df["aqi_rolling_3d"] = df.groupby("City")["aqi_rolling_3d"].bfill()

    print(f"[data_prep] Cleaning complete. Cleaned shape: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct feature matrix X and target y:
    - One-hot encodes 'City'
    - Core pollutants: PM2.5, PM10, NO2, SO2, CO, O3
    - Temporal features: month, day_of_week
    - Seasonal one-hot encoding
    - Lag features: aqi_lag_1, aqi_rolling_3d
    Returns:
        X: Feature DataFrame
        y: Target Series (AQI)
        feature_names: List of column names in X
    """
    df_feat = df.copy()

    # One-hot encode City
    city_dummies = pd.get_dummies(df_feat["City"], prefix="city", dtype=float)

    # One-hot encode Season
    season_dummies = pd.get_dummies(df_feat["season"], prefix="season", dtype=float)

    # Core continuous features
    num_features = POLLUTANT_COLS + ["month", "day_of_week", "aqi_lag_1", "aqi_rolling_3d"]

    # Combine into X
    X = pd.concat([df_feat[num_features], city_dummies, season_dummies], axis=1)
    y = df_feat["AQI"]

    feature_names = list(X.columns)
    return X, y, feature_names


def process_and_save_clean_data(raw_path: str = RAW_CSV_PATH, out_path: str = CLEAN_CSV_PATH) -> pd.DataFrame:
    """End-to-end processing pipeline saving to clean_data.csv."""
    download_dataset_if_needed(raw_path)
    raw_df = load_raw_data(raw_path)
    clean_df = clean_air_quality_data(raw_df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"[data_prep] Saved cleaned dataset to {out_path} ({len(clean_df)} rows).")
    return clean_df


if __name__ == "__main__":
    df = process_and_save_clean_data()
    X, y, f_names = build_features(df)
    print(f"[data_prep] Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    print(f"[data_prep] Feature columns ({len(f_names)}): {f_names}")
