
"""
CSEP Experiment Runner
======================

Compares:
1. Ridge feature selection + SMOTE (baseline)
2. Stability-Selected Features (SSF) + SMOTE
3. SSF + Anomaly-Validated Oversampling (AVO)

The supplied UNSW-NB15 test set is never used to fit preprocessing,
select features, oversample, or train the classifier.

Examples:
    python experiments/run_csep_experiments.py --smoke
    python experiments/run_csep_experiments.py --target binary
    python experiments/run_csep_experiments.py --target multiclass

Smoke mode uses a reproducible, stratified subset of the data.
Its metrics are for debugging, not final research claims.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

# Add the project root BEFORE importing csep.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

from csep.ssf import StableFeatureSelector
from csep.avo import AnomalyValidatedOversampler
from csep.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "experiments" / "results"

RANDOM_STATE = 42
N_SELECTED_FEATURES = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible CSEP experiments."
    )

    parser.add_argument(
        "--target",
        choices=["binary", "multiclass"],
        default="binary",
        help="Prediction target. Defaults to binary for the initial smoke test.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use small stratified samples to test the pipeline.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional maximum number of training rows; 0 means all rows.",
    )

    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Optional maximum number of test rows; 0 means all rows.",
    )

    parser.add_argument(
        "--ssf-folds",
        type=int,
        default=3,
        help="Number of folds used by SSF. Use 2 or 3 for initial testing.",
    )

    return parser.parse_args()


def load_raw_data():
    """Load the original dataset, not previously oversampled arrays."""
    train_path = RAW_DIR / "UNSW_NB15_training-set.csv"
    test_path = RAW_DIR / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected raw files:\n"
            f"  {train_path}\n"
            f"  {test_path}\n"
            "Check the data/raw directory."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Raw training rows: {len(train_df):,}")
    print(f"Raw test rows:     {len(test_df):,}")

    return train_df, test_df



def stratified_subset(X, y, max_rows, random_state=42):
    """Create a reproducible, approximately stratified subset."""
    import numpy as np
    import pandas as pd

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if max_rows <= 0 or len(X) <= max_rows:
        return X, y

    rng = np.random.default_rng(random_state)
    labels = y.to_numpy()
    selected_indices = []

    # Sample each class separately to preserve class representation.
    classes, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    allocations = np.floor(proportions * max_rows).astype(int)

    # Give remaining slots to classes with the largest fractional remainders.
    remaining = max_rows - allocations.sum()
    remainders = proportions * max_rows - allocations

    for idx in np.argsort(-remainders)[:remaining]:
        allocations[idx] += 1

    for class_label, allocation in zip(classes, allocations):
        class_indices = np.flatnonzero(labels == class_label)
        n = min(int(allocation), len(class_indices))

        if n > 0:
            chosen = rng.choice(class_indices, size=n, replace=False)
            selected_indices.extend(chosen.tolist())

    selected_indices = np.array(sorted(selected_indices), dtype=int)

    return (
        X.iloc[selected_indices].reset_index(drop=True),
        y.iloc[selected_indices].reset_index(drop=True),
    )

    """Take a reproducible subset while approximately preserving class ratios."""
    if max_rows <= 0 or len(X) <= max_rows:
        return X.copy(), y.copy()

    if max_rows < y.nunique():
        raise ValueError(
            "max_rows must be at least the number of target classes."
        )

    # Include at least one example from each class.
    fractions = max_rows / len(X)

    sampled_indices = (
        pd.DataFrame({"label": y, "index": np.arange(len(y))})
        .groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(
                    1,
                    min(
                        len(group),
                        int(round(len(group) * fractions)),
                    ),
                ),
                random_state=seed,
            ),
            include_groups=True,
        )["index"]
        .to_numpy()
    )

    # Rounding can make the sample slightly larger than requested.
    if len(sampled_indices) > max_rows:
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(
            sampled_indices, size=max_rows, replace=False
        )

    sampled_indices = np.sort(sampled_indices)

    return (
        X.iloc[sampled_indices].copy(),
        y.iloc[sampled_indices].copy(),
    )


def make_features(train_df, test_df, target):
    """
    Create aligned, numeric feature tables.

    Fit imputation statistics on training rows only.
    Encode categories from training rows only; unseen test categories
    do not introduce new columns.
    """
    target_column = "label" if target == "binary" else "attack_cat"

    if target_column not in train_df or target_column not in test_df:
        raise KeyError(f"Missing target column: {target_column}")

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    # Do not use identifiers or target information as predictors.
    excluded = {"id", "label", "attack_cat"}

    X_train = train_df.drop(
        columns=[c for c in excluded if c in train_df.columns]
    ).copy()

    X_test = test_df.drop(
        columns=[c for c in excluded if c in test_df.columns]
    ).copy()

    categorical_columns = [
        col
        for col in ["proto", "service", "state"]
        if col in X_train.columns
    ]

    # Fill missing categorical values without learning from the test set.
    for col in categorical_columns:
        X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
        X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

    # Numeric columns are explicitly coerced and imputed using train medians.
    numeric_columns = [
        col for col in X_train.columns if col not in categorical_columns
    ]

    for col in numeric_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")

    train_numeric = pd.DataFrame(
        imputer.fit_transform(X_train[numeric_columns]),
        columns=numeric_columns,
        index=X_train.index,
    )

    test_numeric = pd.DataFrame(
        imputer.transform(X_test[numeric_columns]),
        columns=numeric_columns,
        index=X_test.index,
    )

    # One-hot encode separately, then align test columns to training columns.
    train_categorical = pd.get_dummies(
        X_train[categorical_columns],
        dtype=float,
    )

    test_categorical = pd.get_dummies(
        X_test[categorical_columns],
        dtype=float,
    )

    test_categorical = test_categorical.reindex(
        columns=train_categorical.columns,
        fill_value=0,
    )

    X_train_ready = pd.concat(
        [train_numeric, train_categorical], axis=1
    ).astype(float)

    X_test_ready = pd.concat(
        [test_numeric, test_categorical], axis=1
    ).astype(float)

    # Ensure identical feature order and finite values.
    X_test_ready = X_test_ready.reindex(
        columns=X_train_ready.columns, fill_value=0
    )

    if not np.isfinite(X_train_ready.to_numpy()).all():
        raise ValueError("Training features contain non-finite values.")

    if not np.isfinite(X_test_ready.to_numpy()).all():
        raise ValueError("Test features contain non-finite values.")

    # Drop rows with missing target labels, preserving feature/label alignment.
    valid_train = y_train.notna()
    valid_test = y_test.notna()

    X_train_ready = X_train_ready.loc[valid_train].reset_index(drop=True)
    y_train = y_train.loc[valid_train].reset_index(drop=True)

    X_test_ready = X_test_ready.loc[valid_test].reset_index(drop=True)
    y_test = y_test.loc[valid_test].reset_index(drop=True)

    return X_train_ready, y_train, X_test_ready, y_test


def ridge_select_features(X_train, y_train, n_features=10):
    """Select features using Ridge coefficients fitted only on training data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ridge = RidgeClassifier(
        alpha=1.0,
        class_weight="balanced",
    )
    ridge.fit(X_scaled, y_train)

    coefficients = np.asarray(ridge.coef_)

    if coefficients.ndim == 1:
        importance = np.abs(coefficients)
    else:
        importance = np.mean(np.abs(coefficients), axis=0)

    ranking = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": importance,
        }
    ).sort_values(
        "importance", ascending=False, kind="stable"
    )

    selected = ranking["feature"].head(
        min(n_features, X_train.shape[1])
    ).tolist()

    return selected, ranking


def fit_classifier(X_train, y_train):
    """Train the same downstream classifier for every experiment."""
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_test, y_test, selected_features):
    """Calculate classification and probability-quality metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Use the model's explicit class ordering for probability columns.
    classes = np.asarray(model.classes_)
    class_to_index = {label: i for i, label in enumerate(classes)}

    missing_classes = set(np.unique(y_test)) - set(classes)
    if missing_classes:
        raise ValueError(
            f"{name}: test contains classes absent from training: "
            f"{missing_classes}"
        )

    # The project metrics expect sorted class labels.
    sorted_classes = np.sort(classes)
    probabilities_sorted = probabilities[
        :, [class_to_index[label] for label in sorted_classes]
    ]

    metrics = classification_metrics(y_test, predictions)
    metrics["ece"] = expected_calibration_error(
        y_test, probabilities_sorted, n_bins=15
    )
    metrics["brier_score"] = multiclass_brier_score(
        y_test, probabilities_sorted
    )

    metrics["experiment"] = name
    metrics["n_test_samples"] = int(len(y_test))
    metrics["n_selected_features"] = int(len(selected_features))
    metrics["selected_features"] = list(selected_features)

    return metrics, predictions, probabilities_sorted


def run_experiment(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    method,
    ssf_folds,
):
    """Run one feature-selection and resampling configuration."""
    print(f"\n{'-' * 70}")
    print(f"Experiment: {name}")
    print(f"{'-' * 70}")

    start = time.time()

    if method == "ridge_smote":
        selected, ranking = ridge_select_features(
            X_train, y_train, N_SELECTED_FEATURES
        )
        X_fit = X_train[selected].copy()
        X_eval = X_test[selected].copy()

        sampler = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(
            X_fit, y_train
        )

    elif method in {"ssf_smote", "ssf_avo"}:
        selector = StableFeatureSelector(
            n_features=N_SELECTED_FEATURES,
            n_splits=ssf_folds,
            random_state=RANDOM_STATE,
        )

        print("Fitting SSF; this may take some time...")
        selector.fit(X_train, y_train)

        selected = selector.selected_features_
        ranking = selector.report_.feature_table.copy()

        X_fit = selector.transform(X_train)
        X_eval = selector.transform(X_test)

        if method == "ssf_smote":
            sampler = SMOTE(random_state=RANDOM_STATE)
        else:
            # Reduce k for small classes without inventing examples.
            counts = pd.Series(y_train).value_counts()
            min_count = int(counts.min())

            if min_count < 2:
                raise ValueError(
                    "AVO requires at least two genuine examples per class."
                )

            sampler = AnomalyValidatedOversampler(
                k_neighbors=min(5, min_count - 1),
                random_state=RANDOM_STATE,
            )

        X_resampled, y_resampled = sampler.fit_resample(
            X_fit, y_train
        )

        # Save the SSF report for analysis.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(
            OUTPUT_DIR / f"{name}_feature_report.csv",
            index=False,
        )

        if method == "ssf_avo":
            with open(
                OUTPUT_DIR / f"{name}_oversampling_report.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "before": sampler.class_counts_before_,
                        "after": sampler.class_counts_after_,
                        "classes": sampler.class_reports_,
                    },
                    file,
                    indent=2,
                )

    else:
        raise ValueError(f"Unknown experiment method: {method}")

    print(f"Selected features: {len(selected)}")
    print(f"Rows before resampling: {len(y_train):,}")
    print(f"Rows after resampling:  {len(y_resampled):,}")

    model = fit_classifier(X_resampled, y_resampled)

    metrics, predictions, probabilities = evaluate_model(
        name,
        model,
        X_eval,
        y_test,
        selected,
    )

    metrics["training_seconds"] = round(time.time() - start, 2)
    print(json.dumps(metrics, indent=2, default=str))

    return metrics


def main():
    args = parse_args()

    if args.ssf_folds < 2:
        raise ValueError("--ssf-folds must be at least 2.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_raw_data()

    X_train, y_train, X_test, y_test = make_features(
        train_df, test_df, args.target
    )

    if args.smoke:
        # Small, reproducible smoke-test limits.
        train_limit = args.max_train_rows or 8_000
        test_limit = args.max_test_rows or 3_000
    else:
        train_limit = args.max_train_rows
        test_limit = args.max_test_rows

    X_train, y_train = stratified_subset(
        X_train, y_train, train_limit, RANDOM_STATE
    )

    X_test, y_test = stratified_subset(
        X_test, y_test, test_limit, RANDOM_STATE + 1
    )

    print(f"\nTarget: {args.target}")
    print(f"Training rows used: {len(X_train):,}")
    print(f"Test rows used:     {len(X_test):,}")
    print(f"Encoded features:   {X_train.shape[1]:,}")
    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nTest class counts:")
    print(y_test.value_counts().sort_index().to_string())

    results = []

    experiments = [
        ("baseline_ridge_smote", "ridge_smote"),
        ("ssf_smote", "ssf_smote"),
        ("ssf_avo", "ssf_avo"),
    ]

    for name, method in experiments:
        metrics = run_experiment(
            name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            method=method,
            ssf_folds=args.ssf_folds,
        )
        results.append(metrics)

        # Save progress after each experiment in case a later run fails.
        pd.DataFrame(results).to_json(
            OUTPUT_DIR / "csep_experiment_metrics.json",
            orient="records",
            indent=2,
        )

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "experiment": result["experiment"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "ece": result["ece"],
                "brier_score": result["brier_score"],
                "training_seconds": result["training_seconds"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "csep_experiment_summary.csv",
        index=False,
    )

    print("\nFinal experiment summary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
