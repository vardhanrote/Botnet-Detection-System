
"""
Agreement-gated verdict for CyberAgent.

Expected score ranges:
- classifier_confidence: [0, 1], calibrated maximum class probability
- attribution_agreement: [0, 1]
- anomaly_score: [0, 1], with higher meaning more anomalous
- class_plausibility: [0, 1], with higher meaning more plausible

Do not pass raw Isolation Forest score_samples() directly as anomaly_score.
First calibrate it to a percentile or normalized score using training-only
reference data, with the direction explicitly set so higher = more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class VerdictThresholds:
    min_auto_confidence: float = 0.85
    min_auto_agreement: float = 0.65
    max_auto_anomaly: float = 0.70
    min_auto_plausibility: float = 0.60

    low_confidence: float = 0.55
    high_anomaly: float = 0.80
    low_plausibility: float = 0.40


def _validate_score(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def route_alert(
    classifier_confidence: float,
    attribution_agreement: float,
    anomaly_score: float,
    class_plausibility: float,
    thresholds: VerdictThresholds | None = None,
) -> dict:
    """Return a transparent, rule-based investigation routing decision."""

    t = thresholds or VerdictThresholds()

    confidence = _validate_score(
        "classifier_confidence", classifier_confidence
    )
    agreement = _validate_score(
        "attribution_agreement", attribution_agreement
    )
    anomaly = _validate_score("anomaly_score", anomaly_score)
    plausibility = _validate_score(
        "class_plausibility", class_plausibility
    )

    reasons = []

    # Possible novel threat means the classifier is uncertain but the
    # sample is sufficiently unusual or inconsistent with known classes.
    if (
        confidence < t.low_confidence
        and (
            anomaly >= t.high_anomaly
            or plausibility < t.low_plausibility
        )
    ):
        verdict = "POSSIBLE_NOVEL_THREAT"
        reasons.append(
            "Low classifier confidence with high anomaly or low "
            "class plausibility; analyst validation is required."
        )

    # Auto-confirm only when all four signals pass the gate.
    elif (
        confidence >= t.min_auto_confidence
        and agreement >= t.min_auto_agreement
        and anomaly <= t.max_auto_anomaly
        and plausibility >= t.min_auto_plausibility
    ):
        verdict = "AUTO_CONFIRM"
        reasons.append(
            "All configured confidence, agreement, anomaly, and "
            "class-plausibility criteria passed."
        )

    else:
        verdict = "ANALYST_REVIEW"
        reasons.append(
            "At least one confidence-gate criterion was not satisfied "
            "or the evidence signals conflict."
        )

    return {
        "verdict": verdict,
        "scores": {
            "classifier_confidence": confidence,
            "attribution_agreement": agreement,
            "anomaly_score": anomaly,
            "class_plausibility": plausibility,
        },
        "thresholds": asdict(t),
        "reasons": reasons,
        "requires_human_review": verdict != "AUTO_CONFIRM",
        "interpretation": (
            "This is a model-routing decision, not proof that a real "
            "security incident occurred."
        ),
    }
