
import pytest

from csep.verdict_gate import route_alert


def test_high_confidence_consistent_alert_passes_gate():
    result = route_alert(
        classifier_confidence=0.96,
        attribution_agreement=0.90,
        anomaly_score=0.10,
        class_plausibility=0.95,
    )

    assert result["verdict"] == "AUTO_CONFIRM"
    assert result["requires_human_review"] is False


def test_uncertain_anomalous_alert_is_flagged():
    result = route_alert(
        classifier_confidence=0.30,
        attribution_agreement=0.50,
        anomaly_score=0.95,
        class_plausibility=0.20,
    )

    assert result["verdict"] == "POSSIBLE_NOVEL_THREAT"
    assert result["requires_human_review"] is True


def test_conflicting_signals_require_review():
    result = route_alert(
        classifier_confidence=0.92,
        attribution_agreement=0.20,
        anomaly_score=0.10,
        class_plausibility=0.95,
    )

    assert result["verdict"] == "ANALYST_REVIEW"
    assert result["requires_human_review"] is True


def test_invalid_score_is_rejected():
    with pytest.raises(ValueError):
        route_alert(
            classifier_confidence=1.5,
            attribution_agreement=0.8,
            anomaly_score=0.2,
            class_plausibility=0.9,
        )
