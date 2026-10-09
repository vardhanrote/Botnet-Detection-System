
import numpy as np

from csep.metrics import (
    classification_metrics,
    expected_calibration_error,
    feature_top_k_jaccard,
    multiclass_brier_score,
    routing_metrics,
)


def test_classification_metrics_include_macro_f1_and_recall():
    result = classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
    )

    assert "macro_f1" in result
    assert "per_class_recall" in result
    assert result["per_class_recall"]["1"] == 1.0


def test_calibration_metrics_for_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
        [0.90, 0.10],
        [0.10, 0.90],
    ])

    assert expected_calibration_error(y, probabilities) >= 0
    assert multiclass_brier_score(y, probabilities) >= 0


def test_top_k_jaccard():
    assert feature_top_k_jaccard(
        ["a", "b", "c"],
        ["a", "b", "d"],
        k=2,
    ) == 1.0

    assert feature_top_k_jaccard(
        ["a", "b"],
        ["c", "d"],
        k=2,
    ) == 0.0


def test_routing_metrics_count_errors_sent_to_review():
    result = routing_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 0, 1, 1],
        verdicts=[
            "AUTO_CONFIRM",
            "ANALYST_REVIEW",
            "AUTO_CONFIRM",
            "POSSIBLE_NOVEL_THREAT",
        ],
    )

    assert result["auto_confirm_coverage"] == 0.5
    assert result["review_rate"] == 0.5
    assert result["errors_routed_to_review"] == 2
    assert result["error_routing_recall"] == 1.0
