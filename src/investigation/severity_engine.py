"""
CyberAgent Severity / Risk Engine

Calculates a transparent investigation severity score
from existing ML evidence.

This is NOT a probability of attack.

It is an analyst-prioritization score from 0 to 100.
"""


class SeverityEngine:

    BASE_THREAT_SCORES = {
        "Normal": 0,
        "Generic": 35,
        "Exploits": 70,
        "Fuzzers": 65,
        "DoS": 85,
        "Reconnaissance": 45,
        "Analysis": 40,
        "Backdoor": 90,
        "Shellcode": 90,
        "Worms": 95
    }

    SEVERITY_LEVELS = [
        (80, "Critical"),
        (60, "High"),
        (35, "Medium"),
        (15, "Low"),
        (0, "Informational")
    ]

    def __init__(self):
        pass

    def _safe_float(self, value, default=0.0):

        try:
            if value is None:
                return default

            return float(value)

        except (TypeError, ValueError):
            return default

    def _get_threat_class(self, evidence):

        classification = evidence.get(
            "attack_classification",
            {}
        )

        rf = classification.get(
            "random_forest",
            {}
        )

        predicted_class = rf.get(
            "predicted_class"
        )

        if predicted_class:
            return predicted_class

        two_stream = classification.get(
            "two_stream",
            {}
        )

        return two_stream.get(
            "multiclass_prediction",
            "Normal"
        )

    def _get_evidence_score(self, evidence):

        evidence_strength = evidence.get(
            "evidence_strength",
            {}
        )

        return self._safe_float(
            evidence_strength.get("score"),
            0
        )

    def calculate_score(self, evidence):

        threat_class = self._get_threat_class(
            evidence
        )

        base_score = self.BASE_THREAT_SCORES.get(
            threat_class,
            25
        )

        binary = evidence.get(
            "binary_detection",
            {}
        )

        classification = evidence.get(
            "attack_classification",
            {}
        )

        rf = classification.get(
            "random_forest",
            {}
        )

        two_stream = classification.get(
            "two_stream",
            {}
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {}
        )

        consistency = evidence.get(
            "model_consistency",
            {}
        )

        attack_probability = self._safe_float(
            binary.get("attack_probability"),
            0
        )

        rf_confidence = self._safe_float(
            rf.get("confidence"),
            0
        )

        two_stream_confidence = self._safe_float(
            two_stream.get(
                "multiclass_confidence"
            ),
            0
        )

        is_anomaly = bool(
            anomaly.get("is_anomaly", False)
        )

        agreement_status = consistency.get(
            "agreement_status",
            "Disagreement"
        )

        evidence_score = self._get_evidence_score(
            evidence
        )

        # --------------------------------------------------
        # Evidence contributions
        # --------------------------------------------------

        binary_component = (
            attack_probability * 15
        )

        confidence_component = (
            (
                rf_confidence +
                two_stream_confidence
            ) / 2
        ) * 15

        anomaly_component = (
            15 if is_anomaly else 0
        )

        agreement_component = (
            10
            if agreement_status == "Agreement"
            else 0
        )

        evidence_component = (
            evidence_score / 100
        ) * 15

        # --------------------------------------------------
        # Base threat contribution
        # --------------------------------------------------

        base_component = base_score * 0.30

        # --------------------------------------------------
        # Raw score
        # --------------------------------------------------

        raw_score = (
            base_component
            + binary_component
            + confidence_component
            + anomaly_component
            + agreement_component
            + evidence_component
        )

        # --------------------------------------------------
        # Normal traffic protection
        # --------------------------------------------------

        if threat_class == "Normal":

            raw_score *= 0.50

        # --------------------------------------------------
        # If binary detector says normal, reduce severity.
        # --------------------------------------------------

        if binary.get("prediction") == "Normal":

            raw_score *= 0.65

        score = max(
            0,
            min(
                100,
                raw_score
            )
        )

        return round(score, 2)

    def get_level(self, score):

        for threshold, level in self.SEVERITY_LEVELS:

            if score >= threshold:
                return level

        return "Informational"

    def generate_recommendations(
        self,
        evidence,
        threat_class,
        severity_level
    ):

        recommendations = []

        binary = evidence.get(
            "binary_detection",
            {}
        )

        classification = evidence.get(
            "attack_classification",
            {}
        )

        rf = classification.get(
            "random_forest",
            {}
        )

        two_stream = classification.get(
            "two_stream",
            {}
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {}
        )

        consistency = evidence.get(
            "model_consistency",
            {}
        )

        if severity_level in [
            "Critical",
            "High"
        ]:

            recommendations.append(
                "Prioritize this event for analyst review."
            )

        if binary.get("prediction") == "Attack":

            recommendations.append(
                "Review the source and destination "
                "communication associated with the event."
            )

        if anomaly.get("is_anomaly"):

            recommendations.append(
                "Inspect the traffic for behavior that "
                "deviates from the learned normal profile."
            )

        if consistency.get(
            "agreement_status"
        ) == "Disagreement":

            recommendations.append(
                "Review the disagreement between "
                "the supervised classifiers before "
                "taking a definitive action."
            )

        if threat_class == "DoS":

            recommendations.append(
                "Check traffic volume and the availability "
                "of the targeted service."
            )

        elif threat_class == "Reconnaissance":

            recommendations.append(
                "Review scanning and probing activity "
                "across network services."
            )

        elif threat_class == "Exploits":

            recommendations.append(
                "Investigate the targeted service for "
                "possible vulnerability exploitation."
            )

        elif threat_class == "Backdoor":

            recommendations.append(
                "Review unusual remote-access and "
                "outbound communication patterns."
            )

        elif threat_class == "Worms":

            recommendations.append(
                "Check whether similar traffic is "
                "spreading across multiple hosts."
            )

        elif threat_class == "Shellcode":

            recommendations.append(
                "Inspect the targeted host for "
                "suspicious payload-related activity."
            )

        if not recommendations:

            recommendations.append(
                "Continue monitoring the traffic "
                "and compare it with normal behavior."
            )

        return recommendations

    def evaluate(self, evidence):

        threat_class = self._get_threat_class(
            evidence
        )

        score = self.calculate_score(
            evidence
        )

        severity_level = self.get_level(
            score
        )

        recommendations = (
            self.generate_recommendations(
                evidence,
                threat_class,
                severity_level
            )
        )

        return {
            "threat_class": threat_class,

            "severity_score": score,

            "severity_level": severity_level,

            "score_interpretation": (
                "This score represents investigation "
                "priority based on available evidence. "
                "It is not an attack probability."
            ),

            "recommendations": recommendations
        }