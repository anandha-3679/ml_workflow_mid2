from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.features.pipeline import build_preprocessor


def test_build_preprocessor_fit_transform():
    data_path = Path("heartdisease_raw.data")
    df = load_raw_heart_disease_data(data_path)
    x, y = get_feature_and_target_matrices(df)

    # Train / test split to check for leak-free fitting
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # Fit strictly on train set
    x_train_transformed = preprocessor.fit_transform(x_train)
    # Transform test set without refitting
    x_test_transformed = preprocessor.transform(x_test)

    assert isinstance(x_train_transformed, np.ndarray)
    assert isinstance(x_test_transformed, np.ndarray)

    # Check there are no remaining NaNs
    assert not np.isnan(x_train_transformed).any()
    assert not np.isnan(x_test_transformed).any()

    # Transformed columns > original features due to one-hot encoding
    assert x_train_transformed.shape[1] > x_train.shape[1]
    assert x_train_transformed.shape[0] == len(x_train)
    assert x_test_transformed.shape[0] == len(x_test)
    assert x_train_transformed.shape[1] == x_test_transformed.shape[1]
