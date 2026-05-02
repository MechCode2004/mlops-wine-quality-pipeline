import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from train import preprocess_data  # noqa: E402
from utils import validate_data  # noqa: E402


def test_validate_data_success():
    data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "quality": [5, 6, 7],
        }
    )

    validate_data(data, "quality")


def test_preprocess_data_creates_binary_target():
    data = pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.8, 10.3],
            "volatile acidity": [0.7, 0.88, 0.32],
            "citric acid": [0.0, 0.45, 0.32],
            "quality": [5, 6, 7],
        }
    )

    X, y = preprocess_data(data, "quality")

    assert "quality" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)
