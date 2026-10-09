
"""
SSF: Stability-Selected Features

Purpose:
1. Measure feature importance on multiple held-out folds of the training data.
2. Measure how consistently each feature ranks highly.
3. Penalize redundancy between highly correlated features.
4. Select a stable, less-redundant feature subset.

Important:
Call fit() only on the training partition of the outer experiment.
Do not fit this selector on the final test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


@dataclass
class SSFReport:
    """Diagnostics describing why SSF selected its features."""

    feature_table: pd.DataFrame
    selected_features: list[str]
    fold_count: int


class StableFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Stability-aware feature selector for classification.

    Stability is estimated using:
    - permutation importance on held-out training folds;
    - the frequency with which a feature appears in the top-ranked set;
    - median importance rank across folds.

    Redundancy is controlled by avoiding the selection of a feature that
    is too strongly correlated with an already-selected feature.

    Parameters
    ----------
    n_features:
        Maximum number of features to select.

    n_splits:
        Number of stratified folds used within the supplied training data.

    top_fraction:
        Fraction of features considered top-ranked in each fold.

    min_selection_frequency:
        Minimum fraction of folds in which a feature must be top-ranked.
        Set to 0.0 to disable this eligibility filter.

    correlation_threshold:
        Absolute Pearson correlation above which two features are treated
        as redundant during the greedy selection step.

    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int = 10,
        n_splits: int = 5,
        top_fraction: float = 0.30,
        min_selection_frequency: float = 0.40,
        correlation_threshold: float = 0.90,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_splits = n_splits
        self.top_fraction = top_fraction
        self.min_selection_frequency = min_selection_frequency
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

    def _as_frame(self, X) -> pd.DataFrame:
        """Preserve feature names and reject invalid input."""
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional matrix.")
            frame = pd.DataFrame(
                array,
                columns=[f"feature_{i}" for i in range(array.shape[1])],
            )

        if frame.empty:
            raise ValueError("X cannot be empty.")

        if frame.columns.duplicated().any():
            raise ValueError("Feature names must be unique.")

        frame = frame.apply(pd.to_numeric, errors="raise")

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "SSF requires finite numeric features. "
                "Impute missing values before fitting SSF."
            )

        return frame

    def fit(self, X, y):
        """Fit the stability selector using training data only."""
        frame = self._as_frame(X)
        labels = np.asarray(y)

        if len(frame) != len(labels):
            raise ValueError("X and y must contain the same number of rows.")

        if self.n_features < 1:
            raise ValueError("n_features must be at least 1.")

        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")

        if not 0 <= self.min_selection_frequency <= 1:
            raise ValueError(
                "min_selection_frequency must be between 0 and 1."
            )

        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1].")

        class_counts = pd.Series(labels).value_counts()
        smallest_class = int(class_counts.min())

        if smallest_class < 2:
            raise ValueError(
                "At least two training examples per class are required "
                "for stratified fold-based feature selection."
            )

        folds = min(self.n_splits, smallest_class)

        if folds < 2:
            raise ValueError("At least two folds are required.")

        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )

        feature_names = list(frame.columns)
        n_columns = len(feature_names)

        # Each row represents one fold's permutation-importance scores.
        fold_importances = np.zeros((folds, n_columns), dtype=float)
        fold_ranks = np.zeros((folds, n_columns), dtype=float)
        top_hits = np.zeros(n_columns, dtype=float)

        for fold_id, (train_idx, validation_idx) in enumerate(
            splitter.split(frame, labels)
        ):
            X_fold_train = frame.iloc[train_idx]
            X_fold_valid = frame.iloc[validation_idx]
            y_fold_train = labels[train_idx]
            y_fold_valid = labels[validation_idx]

            # This model sees only the fold's training partition.
            model = RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                random_state=self.random_state + fold_id,
                n_jobs=-1,
            )
            model.fit(X_fold_train, y_fold_train)

            # Permutation importance is measured on the held-out fold.
            result = permutation_importance(
                model,
                X_fold_valid,
                y_fold_valid,
                scoring="balanced_accuracy",
                n_repeats=5,
                random_state=self.random_state + fold_id,
                n_jobs=-1,
            )

            # Negative permutation scores are possible. Preserve them
            # in the report rather than pretending every feature helps.
            scores = result.importances_mean
            fold_importances[fold_id] = scores

            # Stable ordering: highest importance receives rank 1.
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(n_columns, dtype=float)
            ranks[order] = np.arange(1, n_columns + 1)
            fold_ranks[fold_id] = ranks

            top_count = max(
                1,
                int(np.ceil(self.top_fraction * n_columns)),
            )
            top_hits[order[:top_count]] += 1

        selection_frequency = top_hits / folds
        median_rank = np.median(fold_ranks, axis=0)
        mean_importance = np.mean(fold_importances, axis=0)
        importance_std = np.std(fold_importances, axis=0)

        # A lower median rank is better. Selection frequency rewards
        # consistency. Mean importance breaks ties between similar ranks.
        feature_table = pd.DataFrame(
            {
                "feature": feature_names,
                "selection_frequency": selection_frequency,
                "median_rank": median_rank,
                "mean_permutation_importance": mean_importance,
                "importance_std": importance_std,
            }
        )

        # Rank-based stability score, with a small importance term.
        feature_table["stability_score"] = (
            feature_table["selection_frequency"]
            * (1.0 - (feature_table["median_rank"] - 1) / n_columns)
        )

        feature_table = feature_table.sort_values(
            by=[
                "stability_score",
                "selection_frequency",
                "mean_permutation_importance",
            ],
            ascending=[False, False, False],
            kind="stable",
        ).reset_index(drop=True)

        eligible = feature_table[
            feature_table["selection_frequency"]
            >= self.min_selection_frequency
        ]

        # If the stability threshold is too strict for this dataset,
        # retain the best-ranked candidates rather than returning no
        # features. This fallback is explicitly recorded in the report.
        if eligible.empty:
            eligible = feature_table.copy()
            self.used_frequency_fallback_ = True
        else:
            self.used_frequency_fallback_ = False

        correlation = frame.corr(method="pearson").abs()
        selected: list[str] = []

        # Greedy redundancy control: consider candidates in stability
        # order, rejecting a feature if it duplicates one already chosen.
        for feature in eligible["feature"]:
            too_correlated = any(
                correlation.loc[feature, chosen]
                >= self.correlation_threshold
                for chosen in selected
            )

            if not too_correlated:
                selected.append(feature)

            if len(selected) >= min(self.n_features, n_columns):
                break

        # A dataset can contain more features than the non-redundant
        # candidates. Fill any remaining slots in score order.
        if len(selected) < min(self.n_features, n_columns):
            for feature in feature_table["feature"]:
                if feature not in selected:
                    selected.append(feature)
                if len(selected) >= min(self.n_features, n_columns):
                    break

        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.selected_features_ = selected
        self.n_features_in_ = n_columns
        self.report_ = SSFReport(
            feature_table=feature_table,
            selected_features=selected.copy(),
            fold_count=folds,
        )

        return self

    def transform(self, X):
        """Return only the selected columns, in the selected order."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Call fit() before transform().")

        frame = self._as_frame(X)
        missing = set(self.selected_features_) - set(frame.columns)

        if missing:
            raise ValueError(
                f"Input is missing selected features: {sorted(missing)}"
            )

        return frame.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        """Return the selected feature names for sklearn compatibility."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("Call fit() before requesting feature names.")

        return np.asarray(self.selected_features_, dtype=object)
