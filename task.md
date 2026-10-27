# Architecture Overview
End-to-end Machine Learning pipeline for clinical Heart Disease Risk Prediction:
- Data ingestion & clinical validation
- Scikit-Learn preprocessing pipeline (`ColumnTransformer`)
- Model benchmarking (Logistic Regression baseline vs. Tree Ensembles)
- Diagnostic evaluation prioritizing clinical sensitivity/recall & ROC-AUC
- Model explanation (feature importance / attribution)
- Serialization & inference engine

## Immediate Next Steps
- None (All planned phases completed successfully)

## Completed Tasks
- [x] Initialized project documentation (`README.md`)
- [x] Bootstrapped `.antigravityrules`
- [x] Configured `ruff` errors-only linting and `pytest` defaults (`pyproject.toml`)
- [x] Phase 1: Environment & Dataset Provisioning (`src/data/loader.py` + tests passing)
- [x] Phase 2: Leak-Free Preprocessing Pipeline (`src/features/pipeline.py` + tests passing)
- [x] Phase 3: Model Training, Cross-Validation & Clinical Evaluation (`src/models/` + tests passing)
- [x] Phase 4: Model Explainability & Interpretability (`src/models/explain.py` + tests passing)
- [x] Phase 5: Artifact Serialization & Inference Engine (`src/inference/` + CLI runner)
- [x] Phase 6: Quality Gates (`ruff check --quiet`, `pytest -q`, and `python main.py` execution verified)
