
"""
AVO: Anomaly-Validated Oversampling

For each minority class:
1. Generate synthetic candidates using SMOTE.
2. Check candidate neighbourhoods against original training data.
3. Fit an Isolation Forest to real training examples of the target class.
4. Accept candidates that pass both checks.

Use only the current training partition. Never oversample validation/test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


class AnomalyValidatedOversampler:
    def __init__(
        self,
        k_neighbors: int = 5,
        locality_threshold: float = 0.60,
        anomaly_quantile: float = 0.99,
        random_state: int = 42,
        n_estimators: int = 200,
    ):
        self.k_neighbors = k_neighbors
        self.locality_threshold = locality_threshold
        self.anomaly_quantile = anomaly_quantile
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit_resample(self, X, y):
        is_frame = isinstance(X, pd.DataFrame)
        columns = list(X.columns) if is_frame else None
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("X and y must have matching rows.")

        if not np.isfinite(X_array).all():
            raise ValueError("AVO requires finite numeric features.")

        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) < 2:
            raise ValueError("AVO requires at least two classes.")

        majority_count = int(counts.max())
        synthetic_X = []
        synthetic_y = []
        self.class_reports_ = {}

        for target_class, class_count in zip(classes, counts):
            class_count = int(class_count)

            # No oversampling is needed for the majority class or
            # classes already at the majority count.
            if class_count >= majority_count:
                continue

            if class_count <= self.k_neighbors:
                raise ValueError(
                    f"Class {target_class!r} has {class_count} examples. "
                    f"SMOTE with k_neighbors={self.k_neighbors} requires "
                    "more target-class examples. Reduce k_neighbors or "
                    "use a larger training sample; do not fabricate data."
                )

            # Generate candidates for one class at a time.
            sampler = SMOTE(
                sampling_strategy={target_class: majority_count},
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
            X_resampled, y_resampled = sampler.fit_resample(
                X_array, y_array
            )

            # imbalanced-learn keeps the original observations first.
            # Candidates are the appended observations for this class.
            candidate_X = np.asarray(X_resampled)[len(X_array):]
            candidate_y = np.asarray(y_resampled)[len(y_array):]

            if len(candidate_X) == 0:
                self.class_reports_[str(target_class)] = {
                    "generated": 0,
                    "accepted": 0,
                }
                continue

            # Fit neighbourhood validation against the ORIGINAL training
            # rows. Synthetic candidates are not used as references.
            neighbour_count = min(self.k_neighbors, len(X_array))
            neighbours = NearestNeighbors(
                n_neighbors=neighbour_count
            ).fit(X_array)

            neighbour_indices = neighbours.kneighbors(
                candidate_X, return_distance=False
            )
            neighbour_labels = y_array[neighbour_indices]

            locality_scores = np.mean(
                neighbour_labels == target_class, axis=1
            )
            locality_ok = (
                locality_scores >= self.locality_threshold
            )

            # The class reference detector sees only genuine training
            # examples from the target class.
            target_X = X_array[y_array == target_class]

            detector = IsolationForest(
                n_estimators=self.n_estimators,
                contamination="auto",
                random_state=self.random_state,
            )
            detector.fit(target_X)

            # IsolationForest.score_samples: higher is more inlier-like.
            real_scores = detector.score_samples(target_X)
            candidate_scores = detector.score_samples(candidate_X)

            # Calibrate the acceptance cutoff from the target class's
            # own training score distribution.
            anomaly_cutoff = float(
                np.quantile(real_scores, 1.0 - self.anomaly_quantile)
            )
            anomaly_ok = candidate_scores >= anomaly_cutoff

            accepted = locality_ok & anomaly_ok

            accepted_X = candidate_X[accepted]
            accepted_y = candidate_y[accepted]

            if len(accepted_X):
                synthetic_X.append(accepted_X)
                synthetic_y.append(accepted_y)

            self.class_reports_[str(target_class)] = {
                "original_class_count": class_count,
                "generated": int(len(candidate_X)),
                "accepted": int(accepted.sum()),
                "rejected_locality": int((~locality_ok).sum()),
                "rejected_anomaly": int((~anomaly_ok).sum()),
                "rejected_both": int((~locality_ok & ~anomaly_ok).sum()),
                "acceptance_rate": float(accepted.mean()),
                "locality_threshold": self.locality_threshold,
                "anomaly_score_cutoff": anomaly_cutoff,
            }

        if synthetic_X:
            X_out = np.vstack([X_array, *synthetic_X])
            y_out = np.concatenate([y_array, *synthetic_y])
        else:
            X_out, y_out = X_array.copy(), y_array.copy()

        if is_frame:
            X_out = pd.DataFrame(X_out, columns=columns)

        self.class_counts_before_ = {
            str(c): int(n) for c, n in zip(*np.unique(y_array, return_counts=True))
        }
        self.class_counts_after_ = {
            str(c): int(n) for c, n in zip(*np.unique(y_out, return_counts=True))
        }

        return X_out, y_out
