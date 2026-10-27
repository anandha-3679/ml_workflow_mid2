from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.models.evaluate import (
    evaluate_clinical_predictions,
    find_optimal_clinical_threshold,
)
from src.models.train import (
    cross_validate_candidates,
    get_model_candidates,
    train_champion_model,
)


def test_model_candidates_structure():
    candidates = get_model_candidates()
    assert "logistic_regression" in candidates
    assert "random_forest" in candidates
    assert "hist_gradient_boosting" in candidates


def test_cross_validation_runs():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    # Fast validation on subset
    x_sub, y_sub = x.iloc[:60], y.iloc[:60]
    results = cross_validate_candidates(x_sub, y_sub, cv_splits=2)

    for model_name, metrics in results.items():
        assert "mean_accuracy" in metrics
        assert "mean_recall" in metrics
        assert "mean_roc_auc" in metrics
        assert 0.0 <= metrics["mean_roc_auc"] <= 1.0


def test_train_and_evaluate_pipeline():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = train_champion_model(x_train, y_train, model_name="logistic_regression")
    probs = pipeline.predict_proba(x_test)[:, 1]

    metrics = evaluate_clinical_predictions(y_test, probs, threshold=0.5)

    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.65
    assert metrics["sensitivity"] > 0.5

    # Threshold optimization
    opt_thresh = find_optimal_clinical_threshold(
        pipeline, x_test, y_test, min_sensitivity=0.80
    )
    assert 0.1 <= opt_thresh <= 0.9
