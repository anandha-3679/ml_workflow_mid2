"""End-to-end clinical heart disease pipeline runner and CLI demonstrator."""

from __future__ import annotations

import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.inference.predict import ClinicalRiskPredictor, save_clinical_pipeline_artifact
from src.models.evaluate import (
    evaluate_clinical_predictions,
    find_optimal_clinical_threshold,
)
from src.models.explain import (
    compute_permutation_importance,
    explain_individual_patient_prediction,
)
from src.models.train import cross_validate_candidates, train_champion_model


def run_pipeline() -> None:
    data_file = Path("heartdisease_raw.data")
    artifact_path = Path("artifacts/heart_disease_pipeline.joblib")

    print("\n=======================================================")
    print("  CLINICAL HEART DISEASE ML PIPELINE EXECUTION")
    print("=======================================================\n")

    # 1. Ingestion & Preprocessing
    print("[1/5] Ingesting and clinically validating raw dataset...")
    df = load_raw_heart_disease_data(data_file)
    x, y = get_feature_and_target_matrices(df)
    print(f"      Total validated patient records: {len(df)}")
    print(f"      Disease prevalence: {y.mean():.1%}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Cross-Validation Benchmarking
    print("\n[2/5] Benchmarking candidate architectures via 5-Fold Stratified CV...")
    cv_scores = cross_validate_candidates(x_train, y_train, cv_splits=5)
    for model_name, metrics in cv_scores.items():
        print(
            f"      - {model_name:23s} | "
            f"ROC-AUC: {metrics['mean_roc_auc']:.3f} | "
            f"Recall: {metrics['mean_recall']:.3f} | "
            f"Accuracy: {metrics['mean_accuracy']:.3f}"
        )

    # 3. Model Training & Clinical Optimization
    print("\n[3/5] Training champion model & optimizing clinical decision threshold...")
    champion = train_champion_model(
        x_train, y_train, model_name="logistic_regression"
    )

    # Optimize threshold targeting >= 85% clinical sensitivity
    optimal_thresh = find_optimal_clinical_threshold(
        champion, x_train, y_train, min_sensitivity=0.85
    )

    test_probs = champion.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_clinical_predictions(
        y_test, test_probs, threshold=optimal_thresh
    )

    print(f"      Optimized Decision Threshold : {optimal_thresh:.2f}")
    print(f"      Test Sensitivity (Recall)   : {test_metrics['sensitivity']:.1%}")
    print(f"      Test Specificity             : {test_metrics['specificity']:.1%}")
    print(f"      Test ROC-AUC                 : {test_metrics['roc_auc']:.3f}")
    print(f"      Test Brier Score             : {test_metrics['brier_score']:.3f}")

    # 4. Model Explainability
    print("\n[4/5] Computing feature importance & local patient attribution...")
    perm_df = compute_permutation_importance(champion, x_test, y_test, n_repeats=5)
    top_3_features = perm_df["feature"].head(3).tolist()
    print(f"      Top 3 Global Risk Drivers: {', '.join(top_3_features)}")

    sample_patient = x_test.iloc[[0]]
    explanation = explain_individual_patient_prediction(
        champion, sample_patient, x_train
    )
    print("      Sample Patient Attribution Breakdown:")
    for factor in explanation["top_contributing_factors"][:3]:
        impact_dir = "Raises Risk" if factor["impact"] > 0 else "Lowers Risk"
        feat = factor["feature"]
        impact = factor["impact"]
        print(f"        * {feat:15s}: {impact:+.3f} ({impact_dir})")

    # 5. Serialization & Standalone Inference Verification
    print("\n[5/5] Serializing pipeline artifact & running inference engine...")
    save_clinical_pipeline_artifact(
        pipeline=champion,
        filepath=artifact_path,
        metadata={
            "model_type": "LogisticRegression",
            "clinical_threshold": optimal_thresh,
            "metrics": test_metrics,
        },
    )
    print(f"      Artifact saved to: {artifact_path}")

    predictor = ClinicalRiskPredictor(artifact_path)
    sample_payload = sample_patient.to_dict(orient="records")[0]
    inference_result = predictor.predict_patient_risk(sample_payload)

    print("\n  Sample Inference Result:")
    print(json.dumps(inference_result, indent=4))
    print("\n=======================================================")
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=======================================================\n")


if __name__ == "__main__":
    run_pipeline()
