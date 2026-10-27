"""Inference package initialization."""
from src.inference.predict import (
    ClinicalRiskPredictor,
    load_clinical_pipeline_artifact,
    save_clinical_pipeline_artifact,
)

__all__ = [
    "ClinicalRiskPredictor",
    "save_clinical_pipeline_artifact",
    "load_clinical_pipeline_artifact",
]
