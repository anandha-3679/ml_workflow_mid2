"""Model artifact persistence and standalone clinical inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def save_clinical_pipeline_artifact(
    pipeline: Pipeline,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize the trained clinical pipeline and associated metadata."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipeline,
        "metadata": metadata or {},
    }
    joblib.dump(payload, path)
    return path


def load_clinical_pipeline_artifact(filepath: str | Path) -> dict[str, Any]:
    """Load serialized clinical model artifact from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path.resolve()}")

    payload: dict[str, Any] = joblib.load(path)
    if "pipeline" not in payload:
        raise ValueError("Invalid artifact format: missing 'pipeline' key")
    return payload


class ClinicalRiskPredictor:
    """Standalone predictor for scoring incoming patient records."""

    def __init__(self, artifact_path: str | Path, default_threshold: float = 0.5):
        artifact = load_clinical_pipeline_artifact(artifact_path)
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metadata: dict[str, Any] = artifact.get("metadata", {})
        self.threshold: float = self.metadata.get(
            "clinical_threshold", default_threshold
        )

    def predict_patient_risk(
        self, patient_data: dict[str, Any] | pd.DataFrame
    ) -> dict[str, Any]:
        """Generate clinical risk prediction with probability and risk tier."""
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data

        prob = float(self.pipeline.predict_proba(df)[0, 1])
        is_high_risk = bool(prob >= self.threshold)

        risk_tier = "High Risk" if is_high_risk else "Low Risk"

        return {
            "risk_tier": risk_tier,
            "risk_probability": round(prob, 4),
            "threshold_applied": self.threshold,
            "actionable_recommendation": (
                "Refer for cardiology consultation and stress diagnostic evaluation."
                if is_high_risk
                else "Routine preventive monitoring and lifestyle management."
            ),
        }
