"""Data loading and validation module for the clinical Heart Disease dataset."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

TARGET_COL = "target"


def load_raw_heart_disease_data(filepath: str | Path) -> pd.DataFrame:
    """Load and clinically validate the raw UCI Heart Disease dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?", "NA", ""],
    )

    # Convert multiclass disease status (0 = no disease, 1-4 = presence)
    # into binary risk indicator: 0 = Low Risk, 1 = High Risk
    df[TARGET_COL] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Enforce numeric dtypes
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clinical validation: flag physiologically implausible values
    # e.g., age must be positive, resting BP must be plausible (> 40)
    df.loc[df["age"] <= 0, "age"] = np.nan
    df.loc[df["trestbps"] <= 40, "trestbps"] = np.nan
    df.loc[df["chol"] <= 40, "chol"] = np.nan

    return df


def get_feature_and_target_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and clinical target label."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in DataFrame")

    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return x, y
