import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.inference.predict import (
    ClinicalRiskPredictor,
    load_clinical_pipeline_artifact,
    save_clinical_pipeline_artifact,
)
from src.models.train import train_champion_model


def test_serialization_and_prediction_roundtrip():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = train_champion_model(x_train, y_train, model_name="logistic_regression")

    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = Path(tmpdir) / "test_pipeline.joblib"
        meta = {"clinical_threshold": 0.42, "version": "1.0.0"}

        # Save artifact
        saved = save_clinical_pipeline_artifact(pipeline, art_path, metadata=meta)
        assert saved.exists()

        # Load artifact directly
        loaded = load_clinical_pipeline_artifact(art_path)
        assert "pipeline" in loaded
        assert loaded["metadata"]["clinical_threshold"] == 0.42

        # Predict using predictor class
        predictor = ClinicalRiskPredictor(art_path)
        sample_dict = x_test.iloc[0].to_dict()
        result = predictor.predict_patient_risk(sample_dict)

        assert "risk_tier" in result
        assert result["risk_tier"] in {"High Risk", "Low Risk"}
        assert 0.0 <= result["risk_probability"] <= 1.0
        assert result["threshold_applied"] == 0.42
        assert "actionable_recommendation" in result
