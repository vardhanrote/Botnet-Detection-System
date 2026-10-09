
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.features.stability_selected_features import (
    StabilitySelectedFeatures,
)


def test_ssf_selects_valid_features_and_preserves_order():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        weights=[0.55, 0.30, 0.15],
        random_state=42,
    )

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=columns)

    selector = StabilitySelectedFeatures(
        estimator=RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
        ),
        max_features=8,
        min_features=4,
        n_splits=3,
        top_n_per_fold=6,
        permutation_repeats=2,
        shap_background_size=20,
        shap_evaluation_size=10,
        random_state=42,
    )

    selector.fit(X, y)
    X_selected = selector.transform(X)

    assert 4 <= X_selected.shape[1] <= 8
    assert list(X_selected.columns) == selector.selected_features_
    assert set(selector.selected_features_).issubset(set(columns))
    assert selector.feature_report_["combined_score"].notna().all()


def test_ssf_does_not_accept_missing_values():
    X = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0],
        "b": [4.0, 3.0, 2.0, 1.0],
    })
    y = np.array([0, 0, 1, 1])

    selector = StabilitySelectedFeatures(
        estimator=RandomForestClassifier(n_estimators=5),
        max_features=2,
        min_features=1,
        n_splits=2,
    )

    try:
        selector.fit(X, y)
    except ValueError as error:
        assert "missing values" in str(error)
    else:
        raise AssertionError("SSF should reject missing values.")
