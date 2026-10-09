
"""
SSF: Stability-Selected Features for CyberAgent CSEP.

The selector:
1. Fits an estimator on stratified folds of the training partition.
2. Measures SHAP and permutation importance on each fold's validation data.
3. Measures feature ranking stability across folds.
4. Penalizes redundant, highly correlated features.
5. Selects the final feature subset.

Important:
Pass only the current training partition to fit().
Never fit this selector on the final test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted


class StabilitySelectedFeatures(BaseEstimator, TransformerMixin):
    """Select stable, useful, and non-redundant numeric features."""

    def __init__(
        self,
        estimator: Any,
        max_features: int = 20,
        min_features: int = 5,
        n_splits: int = 5,
        top_n_per_fold: int = 20,
        correlation_threshold: float = 0.90,
        permutation_repeats: int = 3,
        shap_background_size: int = 40,
        shap_evaluation_size: int = 30,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.max_features = max_features
        self.min_features = min_features
        self.n_splits = n_splits
        self.top_n_per_fold = top_n_per_fold
        self.correlation_threshold = correlation_threshold
        self.permutation_repeats = permutation_repeats
        self.shap_background_size = shap_background_size
        self.shap_evaluation_size = shap_evaluation_size
        self.random_state = random_state

    @staticmethod
    def _as_dataframe(X):
        """Keep feature names when X is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            array = np.asarray(X)
            result = pd.DataFrame(
                array,
                columns=[f"feature_{i}" for i in range(array.shape[1])],
            )

        if result.columns.duplicated().any():
            raise ValueError("Feature names must be unique.")

        if not all(pd.api.types.is_numeric_dtype(t) for t in result.dtypes):
            raise ValueError(
                "SSF expects numeric features. Apply your existing "
                "categorical encoding and preprocessing first."
            )

        if result.isna().any().any():
            raise ValueError(
                "SSF received missing values. Apply your existing "
                "training-fitted imputation before feature selection."
            )

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("SSF received infinite feature values.")

        return result

    @staticmethod
    def _rank_scores(importances):
        """
        Convert importances to comparable scores in [0, 1].
        Higher importance receives a higher score.
        """
        values = np.asarray(importances, dtype=float)
        n_features = len(values)

        if n_features == 1:
            return np.ones(1, dtype=float)

        # Stable descending sort: ties are resolved consistently.
        order = np.argsort(-values, kind="mergesort")
        ranks = np.empty(n_features, dtype=float)
        ranks[order] = np.arange(1, n_features + 1)

        return 1.0 - (ranks - 1.0) / (n_features - 1.0)

    @staticmethod
    def _aggregate_shap_values(shap_values, n_features):
        """Aggregate SHAP magnitudes across rows and output classes."""
        values = getattr(shap_values, "values", shap_values)

        # Some SHAP explainers return one array per class.
        if isinstance(values, list):
            values = np.stack(
                [np.asarray(v) for v in values],
                axis=-1,
            )

        values = np.asarray(values)

        if values.ndim == 2:
            # Samples x features
            importance = np.mean(np.abs(values), axis=0)

        elif values.ndim == 3:
            # Common formats:
            # Samples x features x classes
            # Samples x classes x features
            if values.shape[1] == n_features:
                importance = np.mean(np.abs(values), axis=(0, 2))
            elif values.shape[2] == n_features:
                importance = np.mean(np.abs(values), axis=(0, 1))
            else:
                raise ValueError(
                    "Cannot identify the feature dimension in SHAP values: "
                    f"{values.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported SHAP output dimensions: {values.shape}"
            )

        if len(importance) != n_features:
            raise ValueError("SHAP importance length does not match features.")

        return np.asarray(importance, dtype=float)

    def _shap_importance(self, model, X_train, X_valid):
        """
        Compute model-agnostic SHAP values.

        A small, deterministic background and evaluation sample control
        computation time. The validation sample comes from the current
        training partition's internal fold, never the final test set.
        """
        rng = np.random.default_rng(self.random_state)

        background_size = min(
            self.shap_background_size,
            len(X_train),
        )
        evaluation_size = min(
            self.shap_evaluation_size,
            len(X_valid),
        )

        background_indices = rng.choice(
            len(X_train),
            size=background_size,
            replace=False,
        )
        evaluation_indices = rng.choice(
            len(X_valid),
            size=evaluation_size,
            replace=False,
        )

        background = X_train.iloc[background_indices]
        evaluation = X_valid.iloc[evaluation_indices]

        # Explain class probabilities, not hard predicted class labels.
        explainer = shap.Explainer(
            model.predict_proba,
            background,
            algorithm="permutation",
        )

        # Permutation SHAP requires at least 2 * features + 1 evaluations.
        max_evals = 2 * X_train.shape[1] + 1

        explanation = explainer(
            evaluation,
            max_evals=max_evals,
        )

        return self._aggregate_shap_values(
            explanation,
            n_features=X_train.shape[1],
        )

    def fit(self, X, y):
        """Fit SSF on the supplied training partition only."""
        X_df = self._as_dataframe(X)
        y_array = np.asarray(y)

        if len(X_df) != len(y_array):
            raise ValueError("X and y have different row counts.")

        if X_df.empty or X_df.shape[1] == 0:
            raise ValueError("SSF received an empty feature matrix.")

        if self.max_features < 1:
            raise ValueError("max_features must be at least 1.")

        if not 1 <= self.min_features <= self.max_features:
            raise ValueError(
                "Require 1 <= min_features <= max_features."
            )

        class_counts = pd.Series(y_array).value_counts()
        if len(class_counts) < 2:
            raise ValueError("SSF requires at least two target classes.")

        # Each class must occur in at least two rows for stratified CV.
        smallest_class = int(class_counts.min())
        if smallest_class < 2:
            raise ValueError(
                "At least two training examples per class are required "
                "for stratified stability estimation."
            )

        actual_splits = min(self.n_splits, smallest_class)
        splitter = StratifiedKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        feature_names = list(X_df.columns)
        n_features = len(feature_names)
        fold_shap_scores = []
        fold_permutation_scores = []
        fold_ranks = []
        fold_top_features = []

        for fold_number, (train_idx, valid_idx) in enumerate(
            splitter.split(X_df, y_array),
            start=1,
        ):
            X_fold_train = X_df.iloc[train_idx]
            X_fold_valid = X_df.iloc[valid_idx]
            y_fold_train = y_array[train_idx]
            y_fold_valid = y_array[valid_idx]

            model = clone(self.estimator)
            model.fit(X_fold_train, y_fold_train)

            # SHAP-based feature importance.
            shap_importance = self._shap_importance(
                model,
                X_fold_train,
                X_fold_valid,
            )

            # Independent permutation-importance signal.
            permutation_result = permutation_importance(
                model,
                X_fold_valid,
                y_fold_valid,
                scoring="f1_macro",
                n_repeats=self.permutation_repeats,
                random_state=self.random_state + fold_number,
                n_jobs=1,
            )
            permutation_importance_values = np.maximum(
                permutation_result.importances_mean,
                0.0,
            )

            shap_scores = self._rank_scores(shap_importance)
            permutation_scores = self._rank_scores(
                permutation_importance_values
            )

            # Rank each fold's features using both importance signals.
            combined_fold_score = (
                0.60 * shap_scores
                + 0.40 * permutation_scores
            )

            order = np.argsort(
                -combined_fold_score,
                kind="mergesort",
            )
            ranks = np.empty(n_features, dtype=float)
            ranks[order] = np.arange(1, n_features + 1)

            top_count = min(
                self.top_n_per_fold,
                n_features,
            )

            fold_shap_scores.append(shap_scores)
            fold_permutation_scores.append(permutation_scores)
            fold_ranks.append(ranks)
            fold_top_features.append(set(order[:top_count]))

        shap_matrix = np.vstack(fold_shap_scores)
        permutation_matrix = np.vstack(fold_permutation_scores)
        rank_matrix = np.vstack(fold_ranks)

        # Selection frequency measures how often a feature enters the
        # top-ranked subset across internal folds.
        selection_frequency = np.array([
            np.mean([
                feature_idx in selected
                for selected in fold_top_features
            ])
            for feature_idx in range(n_features)
        ])

        mean_rank = rank_matrix.mean(axis=0)
        rank_std = rank_matrix.std(axis=0)

        # Convert rank variability to a [0, 1] stability score.
        stability_score = 1.0 - (
            rank_std / max(n_features - 1, 1)
        )
        stability_score = np.clip(stability_score, 0.0, 1.0)

        mean_shap_score = shap_matrix.mean(axis=0)
        mean_permutation_score = permutation_matrix.mean(axis=0)

        # Explicitly combine importance, selection frequency, and stability.
        combined_score = (
            0.35 * mean_shap_score
            + 0.25 * mean_permutation_score
            + 0.25 * selection_frequency
            + 0.15 * stability_score
        )

        # Redundancy control: retain the higher-ranked feature when two
        # features have absolute Pearson correlation above the threshold.
        correlation = X_df.corr().abs().fillna(0.0)
        priority_order = np.argsort(
            -combined_score,
            kind="mergesort",
        )

        selected_indices = []
        for candidate in priority_order:
            redundant = any(
                correlation.iloc[candidate, selected]
                > self.correlation_threshold
                for selected in selected_indices
            )

            if not redundant:
                selected_indices.append(int(candidate))

            if len(selected_indices) >= min(
                self.max_features,
                n_features,
            ):
                break

        # If correlation filtering leaves too few features, fill the
        # remaining slots in score order, even if correlated.
        if len(selected_indices) < min(
            self.min_features,
            n_features,
        ):
            for candidate in priority_order:
                candidate = int(candidate)
                if candidate not in selected_indices:
                    selected_indices.append(candidate)
                if len(selected_indices) >= min(
                    self.min_features,
                    self.max_features,
                    n_features,
                ):
                    break

        selected_indices = selected_indices[
            : min(self.max_features, n_features)
        ]

        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = n_features
        self.selected_features_ = [
            feature_names[i] for i in selected_indices
        ]
        self.support_ = np.array([
            name in self.selected_features_
            for name in feature_names
        ])

        self.feature_report_ = pd.DataFrame({
            "feature": feature_names,
            "mean_shap_rank_score": mean_shap_score,
            "mean_permutation_rank_score": mean_permutation_score,
            "selection_frequency": selection_frequency,
            "mean_rank": mean_rank,
            "rank_std": rank_std,
            "stability_score": stability_score,
            "combined_score": combined_score,
            "selected": self.support_,
        }).sort_values(
            ["selected", "combined_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

        return self

    def transform(self, X):
        """Return only the selected features in their original order."""
        check_is_fitted(self, "selected_features_")
        X_df = self._as_dataframe(X)

        missing = set(self.selected_features_) - set(X_df.columns)
        if missing:
            raise ValueError(
                f"Input is missing selected features: {sorted(missing)}"
            )

        # Keep the exact order learned during fit.
        return X_df.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
