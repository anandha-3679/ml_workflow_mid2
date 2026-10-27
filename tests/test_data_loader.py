from pathlib import Path
import pytest
from src.data.loader import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    get_feature_and_target_matrices,
    load_raw_heart_disease_data,
)


def test_load_raw_heart_disease_data_success():
    data_path = Path("heartdisease_raw.data")
    assert data_path.exists(), "Raw dataset file missing"

    df = load_raw_heart_disease_data(data_path)

    # Basic shape and column checks
    assert len(df) > 250
    assert TARGET_COL in df.columns
    assert "num" not in df.columns

    # Verify binary target
    unique_targets = set(df[TARGET_COL].unique())
    assert unique_targets.issubset({0, 1})

    # Verify features are present
    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns


def test_get_feature_and_target_matrices():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    assert len(x) == len(y)
    assert TARGET_COL not in x.columns
    assert y.name == TARGET_COL


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_heart_disease_data("non_existent_dataset.data")
