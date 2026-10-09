
"""Research metrics for CSEP experiments."""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    """Classification metrics, including macro-F1 and per-class recall."""
    labels = sorted(set(y_true) | set(y_pred))

    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class_recall": {
            str(label): float(value)
            for label, value in zip(labels, recalls)
        },
    }


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 15,
) -> float:
    """
    Top-label ECE for multiclass or binary probabilities.

    Lower is better. ECE depends on the binning choice, so report the
    number of bins and also report Brier score and a reliability plot.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes).")

    if probabilities.shape[0] != len(y_true):
        raise ValueError("Probability rows must match y_true.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    predicted_indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    # Assumes the columns of probabilities correspond to sorted classes.
    class_labels = np.unique(y_true)
    if len(class_labels) != probabilities.shape[1]:
        raise ValueError(
            "Probability columns must correspond to every class in y_true."
        )

    predicted_labels = class_labels[predicted_indices]
    correct = (predicted_labels == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for index in range(n_bins):
        lower, upper = bin_edges[index], bin_edges[index + 1]

        if index == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)

        if in_bin.any():
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidence[in_bin].mean()
            ece += in_bin.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def multiclass_brier_score(y_true, probabilities) -> float:
    """
    Multiclass Brier score: mean sum of squared probability errors.

    Lower is better. Probability columns must follow sorted class order.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.unique(y_true)

    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError(
            "Probability shape must be (n_samples, number of classes)."
        )

    one_hot = np.column_stack([
        (y_true == label).astype(float) for label in classes
    ])

    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def feature_top_k_jaccard(
    selected_a: list[str],
    selected_b: list[str],
    k: int = 10,
) -> float:
    """Compare top-k selected feature sets across seeds or folds."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    set_a = set(selected_a[:k])
    set_b = set(selected_b[:k])
    union = set_a | set_b

    if not union:
        return 1.0

    return float(len(set_a & set_b) / len(union))


def routing_metrics(y_true, y_pred, verdicts) -> dict:
    """
    Evaluate whether wrong predictions were sent for human review.

    Verdicts must use:
      AUTO_CONFIRM
      ANALYST_REVIEW
      POSSIBLE_NOVEL_THREAT
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    verdicts = np.asarray(verdicts)

    if not (len(y_true) == len(y_pred) == len(verdicts)):
        raise ValueError("y_true, y_pred and verdicts must have equal length.")

    if len(y_true) == 0:
        raise ValueError("Routing metrics require at least one observation.")

    errors = y_true != y_pred
    auto = verdicts == "AUTO_CONFIRM"
    review = ~auto

    total_errors = int(errors.sum())
    auto_errors = int((errors & auto).sum())
    errors_reviewed = int((errors & review).sum())

    return {
        "n_samples": int(len(y_true)),
        "auto_confirm_coverage": float(auto.mean()),
        "review_rate": float(review.mean()),
        "auto_confirm_error_count": auto_errors,
        "auto_confirm_error_rate": (
            float(auto_errors / auto.sum()) if auto.any() else None
        ),
        "errors_routed_to_review": errors_reviewed,
        "error_routing_recall": (
            float(errors_reviewed / total_errors)
            if total_errors else None
        ),
    }
