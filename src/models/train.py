"""Clinical model training and cross-validation pipelines."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocessor


def get_model_candidates() -> dict[str, Any]:
    """Define candidate estimators across linear and ensemble families."""
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
    }


def build_full_pipeline(estimator: Any) -> Pipeline:
    """Combine preprocessing and estimator into a unified pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def cross_validate_candidates(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """Run stratified cross-validation on all candidate models."""
    candidates = get_model_candidates()
    scoring = ["accuracy", "recall", "precision", "roc_auc"]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    results: dict[str, dict[str, float]] = {}

    for name, estimator in candidates.items():
        pipeline = build_full_pipeline(estimator)
        cv_res = cross_validate(pipeline, x, y, cv=cv, scoring=scoring)
        results[name] = {
            "mean_accuracy": float(np.mean(cv_res["test_accuracy"])),
            "mean_recall": float(np.mean(cv_res["test_recall"])),
            "mean_precision": float(np.mean(cv_res["test_precision"])),
            "mean_roc_auc": float(np.mean(cv_res["test_roc_auc"])),
        }

    return results


def train_champion_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
) -> Pipeline:
    """Train selected model pipeline on the provided training set."""
    candidates = get_model_candidates()
    if model_name not in candidates:
        choices = list(candidates)
        raise ValueError(f"Unknown model_name '{model_name}'. Choose: {choices}")

    pipeline = build_full_pipeline(candidates[model_name])
    pipeline.fit(x, y)
    return pipeline
