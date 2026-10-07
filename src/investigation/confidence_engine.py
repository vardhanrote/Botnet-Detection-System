"""
CyberAgent Investigation Confidence Engine

Calculates a transparent confidence score for the
investigation conclusion.

Important:
This score represents the strength and consistency
of available evidence.

It is NOT an attack probability.
"""

from typing import Any, Dict


class InvestigationConfidenceEngine:
    """
    Calculates investigation confidence using evidence
    already produced by CyberAgent.

    The score combines:
    - Binary detection confidence
    - Random Forest confidence
    - Two-Stream confidence
    - Model agreement
    - Evidence strength
    - Anomaly evidence
    - Threat intelligence support
    """

    def __init__(self):
        # Weights sum to 1.0
        self.weights = {
            "binary": 0.20,
            "classifier": 0.25,
            "agreement": 0.20,
            "evidence": 0.20,
            "anomaly": 0.10,
            "threat_intelligence": 0.05,
        }

    # ---------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ):
        """Keep a value inside the specified range."""

        return max(minimum, min(maximum, value))

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """Safely convert a value to float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------------
    # Binary detection score
    # ---------------------------------------------------------

    def calculate_binary_score(
        self,
        binary_detection: Dict[str, Any],
        threat_class: str,
    ) -> float:
        """
        Calculate confidence from the binary detector.

        For attack classes:
            higher attack probability = stronger evidence.

        For Normal:
            lower attack probability = stronger evidence.
        """

        attack_probability = self._safe_float(
            binary_detection.get("attack_probability", 0.0)
        )

        attack_probability = self._clamp(attack_probability)

        if threat_class == "Normal":
            score = 1.0 - attack_probability
        else:
            score = attack_probability

        return self._clamp(score)

    # ---------------------------------------------------------
    # Classifier confidence
    # ---------------------------------------------------------

    def calculate_classifier_score(
        self,
        classification: Dict[str, Any],
    ) -> float:
        """
        Combine Random Forest and Two-Stream confidence.

        If both confidence values are available,
        their average is used.
        """

        rf = classification.get("random_forest", {})
        two_stream = classification.get("two_stream", {})

        rf_confidence = self._safe_float(
            rf.get("confidence", 0.0)
        )

        two_stream_confidence = self._safe_float(
            two_stream.get("multiclass_confidence", 0.0)
        )

        rf_confidence = self._clamp(rf_confidence)
        two_stream_confidence = self._clamp(two_stream_confidence)

        available_scores = []

        if rf_confidence > 0:
            available_scores.append(rf_confidence)

        if two_stream_confidence > 0:
            available_scores.append(two_stream_confidence)

        if not available_scores:
            return 0.0

        return sum(available_scores) / len(available_scores)

    # ---------------------------------------------------------
    # Model agreement
    # ---------------------------------------------------------

    def calculate_agreement_score(
        self,
        model_consistency: Dict[str, Any],
    ) -> float:
        """
        Agreement between supervised classifiers.

        Agreement gives stronger evidence that the
        predicted class is consistent across models.
        """

        disagreement = model_consistency.get(
            "model_disagreement",
            False,
        )

        if disagreement:
            return 0.0

        return 1.0

    # ---------------------------------------------------------
    # Evidence strength
    # ---------------------------------------------------------

    def calculate_evidence_score(
        self,
        evidence_strength: Dict[str, Any],
    ) -> float:
        """
        Convert the existing evidence strength score
        from 0-100 into 0-1.
        """

        score = self._safe_float(
            evidence_strength.get("score", 0.0)
        )

        score = max(0.0, min(100.0, score))

        return score / 100.0

    # ---------------------------------------------------------
    # Anomaly evidence
    # ---------------------------------------------------------

    def calculate_anomaly_score(
        self,
        anomaly_detection: Dict[str, Any],
    ) -> float:
        """
        Isolation Forest anomaly evidence.

        This is supporting evidence only.

        Anomaly detection does not prove that traffic
        is malicious.
        """

        is_anomaly = anomaly_detection.get(
            "is_anomaly",
            False,
        )

        if is_anomaly:
            return 1.0

        return 0.0

    # ---------------------------------------------------------
    # Threat intelligence support
    # ---------------------------------------------------------

    def calculate_threat_intelligence_score(
        self,
        threat_intelligence: Dict[str, Any],
    ) -> float:
        """
        Calculate a small supporting score based on
        candidate threat-intelligence mappings.

        This deliberately has a low weight because
        ATT&CK mappings from flow-level evidence are
        candidate mappings, not confirmed techniques.
        """

        mappings = threat_intelligence.get(
            "mappings",
            threat_intelligence.get("candidate_mappings", [])
        )

        if not isinstance(mappings, list):
            return 0.0

        if len(mappings) == 0:
            return 0.0

        # More than two mappings should not keep increasing
        # confidence indefinitely.
        return min(len(mappings) / 2.0, 1.0)

    # ---------------------------------------------------------
    # Main calculation
    # ---------------------------------------------------------

    def calculate(
        self,
        binary_detection: Dict[str, Any],
        classification: Dict[str, Any],
        anomaly_detection: Dict[str, Any],
        model_consistency: Dict[str, Any],
        evidence_strength: Dict[str, Any],
        threat_intelligence: Dict[str, Any] | None = None,
        threat_class: str = "Normal",
    ) -> Dict[str, Any]:
        """
        Calculate the complete investigation confidence.
        """

        if threat_intelligence is None:
            threat_intelligence = {}

        binary_score = self.calculate_binary_score(
            binary_detection,
            threat_class,
        )

        classifier_score = self.calculate_classifier_score(
            classification,
        )

        agreement_score = self.calculate_agreement_score(
            model_consistency,
        )

        evidence_score = self.calculate_evidence_score(
            evidence_strength,
        )

        anomaly_score = self.calculate_anomaly_score(
            anomaly_detection,
        )

        threat_intelligence_score = (
            self.calculate_threat_intelligence_score(
                threat_intelligence
            )
        )

        # -----------------------------------------------------
        # Weighted score
        # -----------------------------------------------------

        final_score = (
            binary_score * self.weights["binary"]
            + classifier_score * self.weights["classifier"]
            + agreement_score * self.weights["agreement"]
            + evidence_score * self.weights["evidence"]
            + anomaly_score * self.weights["anomaly"]
            + threat_intelligence_score
            * self.weights["threat_intelligence"]
        )

        final_score = self._clamp(final_score)

        score_100 = round(final_score * 100, 2)

        # -----------------------------------------------------
        # Confidence level
        # -----------------------------------------------------

        if score_100 >= 80:
            level = "High"
        elif score_100 >= 60:
            level = "Moderate"
        elif score_100 >= 40:
            level = "Low"
        else:
            level = "Very Low"

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        interpretation = self._generate_interpretation(
            score_100=score_100,
            binary_score=binary_score,
            classifier_score=classifier_score,
            agreement_score=agreement_score,
            evidence_score=evidence_score,
            anomaly_score=anomaly_score,
            threat_intelligence_score=(
                threat_intelligence_score
            ),
        )

        return {
            "score": score_100,
            "level": level,
            "interpretation": interpretation,
            "components": {
                "binary_detection": round(
                    binary_score * 100,
                    2,
                ),
                "classifier_confidence": round(
                    classifier_score * 100,
                    2,
                ),
                "model_agreement": round(
                    agreement_score * 100,
                    2,
                ),
                "evidence_strength": round(
                    evidence_score * 100,
                    2,
                ),
                "anomaly_evidence": round(
                    anomaly_score * 100,
                    2,
                ),
                "threat_intelligence_support": round(
                    threat_intelligence_score * 100,
                    2,
                ),
            },
            "weights": {
                key: round(value, 2)
                for key, value in self.weights.items()
            },
            "note": (
                "Investigation confidence represents "
                "the strength and consistency of available "
                "evidence. It is not an attack probability."
            ),
        }

    # ---------------------------------------------------------
    # Explanation generation
    # ---------------------------------------------------------

    def _generate_interpretation(
        self,
        score_100: float,
        binary_score: float,
        classifier_score: float,
        agreement_score: float,
        evidence_score: float,
        anomaly_score: float,
        threat_intelligence_score: float,
    ) -> str:
        """
        Generate a human-readable explanation of the score.
        """

        reasons = []

        if binary_score >= 0.70:
            reasons.append(
                "binary detection provides strong support"
            )
        elif binary_score < 0.40:
            reasons.append(
                "binary detection provides limited support"
            )

        if classifier_score >= 0.70:
            reasons.append(
                "classifier confidence is strong"
            )
        elif classifier_score < 0.40:
            reasons.append(
                "classifier confidence is limited"
            )

        if agreement_score >= 1.0:
            reasons.append(
                "the supervised classifiers agree"
            )
        else:
            reasons.append(
                "the supervised classifiers disagree"
            )

        if evidence_score >= 0.70:
            reasons.append(
                "overall evidence strength is strong"
            )
        elif evidence_score < 0.40:
            reasons.append(
                "overall evidence strength is limited"
            )

        if anomaly_score >= 1.0:
            reasons.append(
                "Isolation Forest provides supporting anomaly evidence"
            )
        else:
            reasons.append(
                "Isolation Forest does not provide anomaly evidence"
            )

        if threat_intelligence_score > 0:
            reasons.append(
                "candidate threat-intelligence mappings are available"
            )

        explanation = (
            f"Investigation confidence is {score_100:.2f}/100 "
            f"because "
            + ", ".join(reasons)
            + "."
        )

        return explanation