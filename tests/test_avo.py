
import numpy as np

from sklearn.datasets import make_classification

from csep.avo import AnomalyValidatedOversampler


def test_avo_preserves_original_rows_and_reports_candidates():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.80, 0.20],
        random_state=42,
    )

    sampler = AnomalyValidatedOversampler(
        k_neighbors=3,
        locality_threshold=0.50,
        random_state=42,
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Every original observation must still be present at the start.
    np.testing.assert_allclose(X_resampled[:len(X)], X)
    np.testing.assert_array_equal(y_resampled[:len(y)], y)

    assert len(X_resampled) == len(y_resampled)
    assert "1" in sampler.class_reports_ or "0" in sampler.class_reports_

    for report in sampler.class_reports_.values():
        assert report["accepted"] <= report["generated"]
        assert 0.0 <= report["acceptance_rate"] <= 1.0


def test_avo_does_not_oversample_balanced_data():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        weights=[0.50, 0.50],
        random_state=10,
    )

    sampler = AnomalyValidatedOversampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    assert len(X_resampled) == len(X)
    np.testing.assert_array_equal(y_resampled, y)
