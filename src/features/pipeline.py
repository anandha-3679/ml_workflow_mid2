"""Feature engineering and preprocessing pipelines for clinical tabular data."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a leak-proof Scikit-Learn ColumnTransformer preprocessor."""
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Numerical pipeline: impute missing values with median, then standardize
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
