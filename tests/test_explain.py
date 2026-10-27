from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.models.explain import (
    compute_permutation_importance,
    explain_individual_patient_prediction,
    get_feature_names_from_pipeline,
)
from src.models.train import train_champion_model


def test_explainability_linear_and_tree():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    # Test linear explanation
    linear_pipe = train_champion_model(
        x_train, y_train, model_name="logistic_regression"
    )
    feat_names = get_feature_names_from_pipeline(linear_pipe)
    assert len(feat_names) > len(x.columns)

    patient = x_test.iloc[[0]]
    linear_exp = explain_individual_patient_prediction(linear_pipe, patient, x_train)
    assert "predicted_risk_probability" in linear_exp
    assert len(linear_exp["top_contributing_factors"]) == 5

    # Test permutation importance on random forest
    rf_pipe = train_champion_model(x_train, y_train, model_name="random_forest")
    perm_df = compute_permutation_importance(rf_pipe, x_test, y_test, n_repeats=3)
    assert len(perm_df) == len(x.columns)
    assert "importance_mean" in perm_df.columns

    # Test tree model individual patient explanation
    rf_exp = explain_individual_patient_prediction(rf_pipe, patient, x_train)
    assert "predicted_risk_probability" in rf_exp
    assert len(rf_exp["top_contributing_factors"]) == 5
