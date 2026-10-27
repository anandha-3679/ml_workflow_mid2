"""Model interpretability and explainability module for clinical risk factors."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Retrieve transformed feature names from the fitted preprocessor."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline does not contain a 'preprocessor' step")

    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            enc = transformer.named_steps["encoder"]
            cat_names = enc.get_feature_names_out(cols).tolist()
            feature_names.extend(cat_names)
        else:
            feature_names.extend(cols)

    return feature_names


def compute_permutation_importance(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importances on holdout test set."""
    perm_res = permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )

    df_importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": perm_res.importances_mean,
            "importance_std": perm_res.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    return df_importance.reset_index(drop=True)


def explain_individual_patient_prediction(
    pipeline: Pipeline,
    patient_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> dict[str, Any]:
    """Provide local risk factor attribution for an individual patient."""
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    prob = float(pipeline.predict_proba(patient_df)[0, 1])

    if hasattr(classifier, "coef_"):
        # Linear model explanation via transformed coefficients
        patient_trans = preprocessor.transform(patient_df)[0]
        baseline_trans = preprocessor.transform(baseline_df).mean(axis=0)
        diff = patient_trans - baseline_trans
        coefs = classifier.coef_[0]
        attributions = diff * coefs

        feat_names = get_feature_names_from_pipeline(pipeline)
        breakdown = [
            {"feature": f, "impact": float(attr)}
            for f, attr in sorted(
                zip(feat_names, attributions),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]
    else:
        # Permutation drop surrogate for tree models
        breakdown = []
        base_patient_prob = prob
        for col in patient_df.columns:
            perturbed = patient_df.copy()
            perturbed[col] = baseline_df[col].median()
            pert_prob = float(pipeline.predict_proba(perturbed)[0, 1])
            impact = base_patient_prob - pert_prob
            breakdown.append({"feature": col, "impact": impact})

        breakdown = sorted(breakdown, key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "predicted_risk_probability": prob,
        "top_contributing_factors": breakdown[:5],
    }
