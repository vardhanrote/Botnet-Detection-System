
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification

from csep.ssf import StableFeatureSelector


def make_data():
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=6,
        n_redundant=4,
        n_classes=3,
        weights=[0.55, 0.30, 0.15],
        random_state=42,
    )

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


def test_ssf_selects_requested_number_of_features():
    X, y = make_data()

    selector = StableFeatureSelector(
        n_features=8,
        n_splits=4,
        random_state=42,
    )

    selector.fit(X, y)
    transformed = selector.transform(X)

    assert transformed.shape == (600, 8)
    assert len(set(selector.selected_features_)) == 8


def test_ssf_report_contains_stability_metrics():
    X, y = make_data()
    selector = StableFeatureSelector(n_features=8, n_splits=4)
    selector.fit(X, y)

    report = selector.report_.feature_table

    required = {
        "feature",
        "selection_frequency",
        "median_rank",
        "mean_permutation_importance",
        "importance_std",
        "stability_score",
    }

    assert required.issubset(report.columns)
    assert len(report) == X.shape[1]
    assert report["selection_frequency"].between(0, 1).all()


def test_ssf_reproducible_for_same_seed():
    X, y = make_data()

    first = StableFeatureSelector(
        n_features=8, n_splits=4, random_state=123
    ).fit(X, y)

    second = StableFeatureSelector(
        n_features=8, n_splits=4, random_state=123
    ).fit(X, y)

    assert first.selected_features_ == second.selected_features_


def test_ssf_rejects_non_finite_values():
    X, y = make_data()
    X.loc[0, "feature_0"] = np.nan

    with pytest.raises(ValueError, match="finite numeric features"):
        StableFeatureSelector(n_features=8).fit(X, y)
