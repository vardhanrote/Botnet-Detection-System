"""
CyberAgent Investigation Agent

Evidence-grounded investigation reasoning layer.

This module does NOT independently decide whether traffic is malicious.
Instead, it combines outputs from:

- Binary detector
- Random Forest
- Two-Stream model
- Isolation Forest
- SHAP
- LIME
- Threat profiling
- Severity engine
- MITRE ATT&CK candidate mappings

The goal is to convert model outputs into an analyst-friendly
investigation verdict.

Important:
- Anomaly detection is not proof of maliciousness.
- SHAP/LIME show model reasoning, not causality.
- MITRE mappings are candidate mappings and require corroboration.
"""

from __future__ import annotations

from typing import Any, Dict, List


class InvestigationAgent:
    """
    Evidence-grounded CyberAgent reasoning engine.
    """

    def __init__(self) -> None:
        self.agent_name = "CyberAgent Investigation Agent"
        self.version = "1.0.0"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def investigate(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any] | None = None,
        threat_intelligence: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Perform a complete investigation using existing evidence.
        """

        threat_profile = threat_profile or {}
        threat_intelligence = threat_intelligence or {}

        binary = evidence.get("binary_detection", {})
        classification = evidence.get("attack_classification", {})
        anomaly = evidence.get("anomaly_detection", {})
        consistency = evidence.get("model_consistency", {})
        strength = evidence.get("evidence_strength", {})

        rf = classification.get("random_forest", {})
        two_stream = classification.get("two_stream", {})

        threat_class = self._get_threat_class(
            rf,
            two_stream,
            threat_profile,
        )

        severity = self._get_severity(
            evidence,
            threat_profile,
        )

        findings = self._generate_findings(
            binary=binary,
            rf=rf,
            two_stream=two_stream,
            anomaly=anomaly,
            consistency=consistency,
            strength=strength,
            threat_class=threat_class,
        )

        explainability = self._summarize_explainability(evidence)

        mitre = self._extract_mitre_information(
            threat_intelligence
        )

        reasoning = self._generate_reasoning(
            binary=binary,
            rf=rf,
            two_stream=two_stream,
            anomaly=anomaly,
            consistency=consistency,
            strength=strength,
            threat_class=threat_class,
            severity=severity,
        )

        confidence = self._calculate_agent_confidence(
            evidence=evidence,
            threat_class=threat_class,
        )

        recommendations = self._generate_recommendations(
            threat_class=threat_class,
            severity=severity,
            anomaly=anomaly,
            consistency=consistency,
            mitre=mitre,
        )

        verdict = self._generate_verdict(
            binary=binary,
            threat_class=threat_class,
            severity=severity,
            confidence=confidence,
            consistency=consistency,
        )

        return {
            "agent": {
                "name": self.agent_name,
                "version": self.version,
                "reasoning_type": "evidence-grounded",
            },
            "investigation_verdict": verdict,
            "threat_class": threat_class,
            "severity": severity,
            "agent_confidence": confidence,
            "key_findings": findings,
            "reasoning": reasoning,
            "explainability": explainability,
            "mitre_attack": mitre,
            "recommended_actions": recommendations,
            "limitations": self._limitations(),
        }

    # ------------------------------------------------------------------
    # Threat class
    # ------------------------------------------------------------------

    def _get_threat_class(
        self,
        rf: Dict[str, Any],
        two_stream: Dict[str, Any],
        threat_profile: Dict[str, Any],
    ) -> str:
        """
        Determine the investigation threat class.

        Priority:
        1. Explicit threat profile
        2. RF and Two-Stream agreement
        3. Random Forest
        4. Two-Stream
        """

        profile_class = (
            threat_profile.get("threat_class")
            or threat_profile.get("attack_class")
            or threat_profile.get("predicted_class")
        )

        if profile_class:
            return str(profile_class)

        rf_class = (
            rf.get("predicted_class")
            or rf.get("prediction")
            or "Unknown"
        )

        ts_class = (
            two_stream.get("multiclass_prediction")
            or two_stream.get("predicted_class")
            or "Unknown"
        )

        if rf_class == ts_class:
            return str(rf_class)

        if rf_class != "Unknown":
            return str(rf_class)

        return str(ts_class)

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    def _get_severity(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract severity information from the existing severity engine.
        """

        severity = {}

        if isinstance(threat_profile, dict):
            severity = threat_profile.get("severity", {})

        if not severity:
            severity = evidence.get("severity", {})

        if isinstance(severity, str):
            return {
                "level": severity,
                "score": None,
            }

        return {
            "level": severity.get("level", "Unknown"),
            "score": severity.get("score"),
        }

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def _generate_findings(
        self,
        binary: Dict[str, Any],
        rf: Dict[str, Any],
        two_stream: Dict[str, Any],
        anomaly: Dict[str, Any],
        consistency: Dict[str, Any],
        strength: Dict[str, Any],
        threat_class: str,
    ) -> List[str]:

        findings: List[str] = []

        prediction = str(
            binary.get("prediction", "Unknown")
        ).lower()

        attack_probability = binary.get(
            "attack_probability"
        )

        if prediction == "attack":
            if attack_probability is not None:
                findings.append(
                    "The binary detector identified the traffic "
                    f"as an attack with an attack probability of "
                    f"{float(attack_probability):.2%}."
                )
            else:
                findings.append(
                    "The binary detector identified the traffic "
                    "as an attack."
                )

        elif prediction == "normal":
            findings.append(
                "The binary detector classified the traffic as normal."
            )

        rf_class = rf.get("predicted_class")

        if rf_class:
            rf_confidence = rf.get("confidence")

            if rf_confidence is not None:
                findings.append(
                    f"Random Forest predicted {rf_class} "
                    f"with {float(rf_confidence):.2%} confidence."
                )
            else:
                findings.append(
                    f"Random Forest predicted {rf_class}."
                )

        ts_class = two_stream.get(
            "multiclass_prediction"
        )

        if ts_class:
            ts_confidence = two_stream.get(
                "multiclass_confidence"
            )

            if ts_confidence is not None:
                findings.append(
                    f"Two-Stream predicted {ts_class} "
                    f"with {float(ts_confidence):.2%} confidence."
                )
            else:
                findings.append(
                    f"Two-Stream predicted {ts_class}."
                )

        is_anomaly = anomaly.get("is_anomaly")

        if is_anomaly:
            findings.append(
                "Isolation Forest detected a deviation from "
                "the learned normal traffic distribution."
            )
        else:
            findings.append(
                "Isolation Forest did not flag the sample as anomalous."
            )

        disagreement = consistency.get(
            "model_disagreement"
        )

        if disagreement:
            findings.append(
                "The supervised classifiers disagree on "
                "the specific attack class."
            )
        else:
            findings.append(
                "The supervised classifiers agree on "
                "the predicted attack class."
            )

        evidence_level = strength.get(
            "level"
        )

        if evidence_level:
            findings.append(
                f"Overall evidence strength is {evidence_level}."
            )

        if threat_class != "Unknown":
            findings.append(
                f"The investigation is currently centered on "
                f"the {threat_class} threat category."
            )

        return findings

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _summarize_explainability(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract SHAP/LIME information if available.
        """

        result = {
            "shap": {},
            "lime": {},
        }

        shap_data = evidence.get(
            "shap_local"
        )

        lime_data = evidence.get(
            "lime_local"
        )

        if shap_data is None:
            shap_data = evidence.get(
                "local_shap"
            )

        if lime_data is None:
            lime_data = evidence.get(
                "local_lime"
            )

        if isinstance(shap_data, dict):
            result["shap"] = self._clean_explanation(
                shap_data
            )

        if isinstance(lime_data, dict):
            result["lime"] = self._clean_explanation(
                lime_data
            )

        return result

    def _clean_explanation(
        self,
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Keep explanation output structured without making
        unsupported causal claims.
        """

        result = {}

        for key in [
            "predicted_class",
            "prediction",
            "confidence",
            "top_features",
            "features",
            "explanation",
            "feature_contributions",
        ]:
            if key in explanation:
                result[key] = explanation[key]

        return result

    # ------------------------------------------------------------------
    # MITRE
    # ------------------------------------------------------------------

    def _extract_mitre_information(
        self,
        threat_intelligence: Dict[str, Any],
    ) -> Dict[str, Any]:

        mappings = (
            threat_intelligence.get("mitre_mappings")
            or threat_intelligence.get("mappings")
            or threat_intelligence.get("techniques")
            or []
        )

        guidance = (
            threat_intelligence.get("analyst_guidance")
            or []
        )

        return {
            "candidate_techniques": mappings,
            "analyst_guidance": guidance,
            "mapping_policy": (
                "ATT&CK techniques are candidate mappings based "
                "on flow-level evidence and require corroborating "
                "host, application, authentication, or other "
                "telemetry before being treated as confirmed."
            ),
        }

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self,
        binary: Dict[str, Any],
        rf: Dict[str, Any],
        two_stream: Dict[str, Any],
        anomaly: Dict[str, Any],
        consistency: Dict[str, Any],
        strength: Dict[str, Any],
        threat_class: str,
        severity: Dict[str, Any],
    ) -> List[str]:

        reasoning = []

        prediction = str(
            binary.get("prediction", "Unknown")
        ).lower()

        if prediction == "attack":
            reasoning.append(
                "Step 1: The binary detection stage provides "
                "evidence that the traffic is potentially malicious."
            )
        else:
            reasoning.append(
                "Step 1: The binary detection stage does not "
                "provide sufficient evidence of an attack."
            )

        rf_class = rf.get("predicted_class")
        ts_class = two_stream.get(
            "multiclass_prediction"
        )

        if rf_class == ts_class and rf_class:
            reasoning.append(
                f"Step 2: Random Forest and Two-Stream both "
                f"support the {rf_class} classification."
            )
        else:
            reasoning.append(
                "Step 2: The supervised classifiers provide "
                "different attack-class predictions, reducing "
                "confidence in the exact threat category."
            )

        if anomaly.get("is_anomaly"):
            reasoning.append(
                "Step 3: Isolation Forest identifies the traffic "
                "as a deviation from the learned normal distribution."
            )
        else:
            reasoning.append(
                "Step 3: Isolation Forest does not identify the "
                "sample as anomalous, so anomaly evidence is absent."
            )

        evidence_level = strength.get(
            "level",
            "Unknown"
        )

        reasoning.append(
            f"Step 4: The combined evidence strength is "
            f"{evidence_level}."
        )

        severity_level = severity.get(
            "level",
            "Unknown"
        )

        reasoning.append(
            f"Step 5: The current investigation priority is "
            f"{severity_level} for the {threat_class} category."
        )

        return reasoning

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _calculate_agent_confidence(
        self,
        evidence: Dict[str, Any],
        threat_class: str,
    ) -> Dict[str, Any]:

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

        ts = classification.get(
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

        strength = evidence.get(
            "evidence_strength",
            {}
        )

        score = 0.0
        reasons = []

        # Binary detector
        attack_probability = binary.get(
            "attack_probability"
        )

        if attack_probability is not None:
            attack_probability = float(
                attack_probability
            )

            distance_from_half = abs(
                attack_probability - 0.5
            ) * 2

            score += 25 * distance_from_half

            reasons.append(
                "binary detector confidence"
            )

        # RF confidence
        rf_confidence = rf.get(
            "confidence"
        )

        if rf_confidence is not None:
            score += 20 * float(
                rf_confidence
            )

            reasons.append(
                "Random Forest confidence"
            )

        # Two-Stream confidence
        ts_confidence = ts.get(
            "multiclass_confidence"
        )

        if ts_confidence is not None:
            score += 20 * float(
                ts_confidence
            )

            reasons.append(
                "Two-Stream confidence"
            )

        # Model agreement
        if not consistency.get(
            "model_disagreement",
            False,
        ):
            score += 15
            reasons.append(
                "classifier agreement"
            )

        # Anomaly evidence
        if anomaly.get(
            "is_anomaly",
            False,
        ):
            score += 10
            reasons.append(
                "anomaly evidence"
            )

        # Evidence strength
        evidence_score = strength.get(
            "score"
        )

        if evidence_score is not None:
            score += 10 * (
                float(evidence_score) / 100
            )

            reasons.append(
                "overall evidence strength"
            )

        score = min(
            100.0,
            max(0.0, score)
        )

        if score >= 80:
            level = "High"
        elif score >= 60:
            level = "Medium"
        elif score >= 40:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "score": round(score, 2),
            "level": level,
            "basis": reasons,
            "note": (
                "Agent confidence reflects agreement and strength "
                "of available evidence. It is not a probability "
                "that the attack occurred."
            ),
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        threat_class: str,
        severity: Dict[str, Any],
        anomaly: Dict[str, Any],
        consistency: Dict[str, Any],
        mitre: Dict[str, Any],
    ) -> List[str]:

        recommendations: List[str] = []

        severity_level = severity.get(
            "level",
            "Unknown"
        )

        if severity_level in [
            "Critical",
            "High",
        ]:
            recommendations.append(
                "Prioritize this event for analyst review."
            )

        elif severity_level == "Medium":
            recommendations.append(
                "Review the event and correlate it with "
                "additional security telemetry."
            )

        else:
            recommendations.append(
                "Keep the event under monitoring and "
                "collect additional evidence if available."
            )

        if consistency.get(
            "model_disagreement",
            False,
        ):
            recommendations.append(
                "Investigate the classifier disagreement "
                "before assigning a definitive attack category."
            )

        if anomaly.get(
            "is_anomaly",
            False,
        ):
            recommendations.append(
                "Check surrounding traffic and endpoint telemetry "
                "because the sample differs from learned normal traffic."
            )

        if threat_class not in [
            "Normal",
            "Unknown",
        ]:
            recommendations.append(
                "Correlate the predicted threat category with "
                "host, application, authentication, and network logs."
            )

        if mitre.get(
            "candidate_techniques"
        ):
            recommendations.append(
                "Validate the candidate MITRE ATT&CK techniques "
                "using additional telemetry before treating them "
                "as confirmed."
            )

        recommendations.append(
            "Review SHAP and LIME explanations to understand "
            "which input features influenced the model decision."
        )

        return recommendations

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def _generate_verdict(
        self,
        binary: Dict[str, Any],
        threat_class: str,
        severity: Dict[str, Any],
        confidence: Dict[str, Any],
        consistency: Dict[str, Any],
    ) -> Dict[str, Any]:

        prediction = str(
            binary.get(
                "prediction",
                "Unknown",
            )
        ).lower()

        if prediction == "attack":
            if threat_class not in [
                "Unknown",
                "Normal",
            ]:
                verdict = (
                    f"Potential {threat_class} activity "
                    "requires investigation."
                )
            else:
                verdict = (
                    "Potential malicious activity detected, "
                    "but the exact threat category is uncertain."
                )

        elif prediction == "normal":
            verdict = (
                "No strong evidence of malicious activity was "
                "identified by the binary detector."
            )

        else:
            verdict = (
                "The available evidence is insufficient for "
                "a definitive investigation verdict."
            )

        if consistency.get(
            "model_disagreement",
            False,
        ):
            qualifier = (
                " Classifier disagreement reduces confidence "
                "in the exact attack category."
            )
        else:
            qualifier = ""

        return {
            "verdict": verdict + qualifier,
            "threat_class": threat_class,
            "severity": severity.get(
                "level",
                "Unknown",
            ),
            "confidence_level": confidence.get(
                "level",
                "Very Low",
            ),
            "confidence_score": confidence.get(
                "score",
                0,
            ),
        }

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------

    def _limitations(self) -> List[str]:

        return [
            (
                "The investigation uses flow-level evidence "
                "derived from UNSW-NB15."
            ),
            (
                "DNS features are DNS-derived flow proxies, "
                "not raw DNS packet-level features."
            ),
            (
                "Isolation Forest identifies deviation from "
                "learned normal traffic and does not prove maliciousness."
            ),
            (
                "SHAP and LIME describe model behavior and "
                "should not be interpreted as causal explanations."
            ),
            (
                "MITRE ATT&CK mappings are candidate mappings "
                "and require corroborating telemetry."
            ),
            (
                "Agent confidence is an evidence-strength indicator, "
                "not an attack probability."
            ),
        ]