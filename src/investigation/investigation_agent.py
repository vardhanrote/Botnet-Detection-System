"""
CyberAgent Investigation Agent

Evidence-grounded investigation layer that combines:
- ML detection results
- Model agreement/disagreement
- Anomaly detection
- SHAP/LIME explainability
- Threat profiling
- Severity assessment
- MITRE ATT&CK candidate mappings

The agent does not claim certainty beyond the available evidence.
"""

from typing import Any, Dict, List

from src.investigation.confidence_engine import (
    InvestigationConfidenceEngine,
)


class InvestigationAgent:
    """
    Evidence-grounded investigation agent.

    This component converts outputs from the detection,
    explainability, threat profiling, and threat intelligence
    layers into an analyst-oriented investigation result.
    """

    def __init__(
        self,
        agent_name: str = "CyberAgent Investigation Agent",
        version: str = "1.0.0",
    ):
        self.agent_name = agent_name
        self.version = version

        # Phase 10: Transparent investigation confidence engine
        self.confidence_engine = InvestigationConfidenceEngine()

    # ------------------------------------------------------------------
    # Threat class extraction
    # ------------------------------------------------------------------

    def _get_threat_class(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
    ) -> str:

        # Phase 8 structure
        profile = threat_profile.get("threat_profile", {})

        if profile:
            predicted = profile.get("predicted_threat")

            if predicted:
                return predicted

        # Direct fallback
        predicted = threat_profile.get("predicted_threat")

        if predicted:
            return predicted

        # Evidence fallback
        classification = evidence.get(
            "attack_classification",
            {},
        )

        rf = classification.get(
            "random_forest",
            {},
        )

        predicted = rf.get("predicted_class")

        if predicted:
            return predicted

        return "Unknown"

    # ------------------------------------------------------------------
    # Severity extraction
    # ------------------------------------------------------------------

    def _get_severity(
        self,
        threat_profile: Dict[str, Any],
    ) -> Dict[str, Any]:

        # Phase 8 structure
        severity_data = threat_profile.get(
            "severity_assessment",
            {},
        )

        # Backward compatibility
        if not severity_data:
            severity_data = threat_profile.get(
                "severity",
                {},
            )

        level = severity_data.get(
            "severity_level",
            severity_data.get(
                "level",
                "Unknown",
            ),
        )

        score = severity_data.get(
            "severity_score",
            severity_data.get(
                "score",
            ),
        )

        return {
            "level": level,
            "score": score,
        }

    # ------------------------------------------------------------------
    # Findings generation
    # ------------------------------------------------------------------

    def _generate_findings(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> List[str]:

        findings = []

        binary = evidence.get(
            "binary_detection",
            {},
        )

        classification = evidence.get(
            "attack_classification",
            {},
        )

        rf = classification.get(
            "random_forest",
            {},
        )

        two_stream = classification.get(
            "two_stream",
            {},
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {},
        )

        consistency = evidence.get(
            "model_consistency",
            {},
        )

        evidence_strength = evidence.get(
            "evidence_strength",
            {},
        )

        # Binary detection
        if binary.get("prediction") == "Attack":

            attack_probability = binary.get(
                "attack_probability"
            )

            if attack_probability is not None:

                findings.append(
                    f"Binary detection classified the "
                    f"traffic as attack activity with "
                    f"{attack_probability:.2%} attack probability."
                )

            else:

                findings.append(
                    "Binary detection classified the "
                    "traffic as attack activity."
                )

        else:

            findings.append(
                "Binary detection classified the "
                "traffic as normal traffic."
            )

        # Random Forest
        rf_class = rf.get(
            "predicted_class"
        )

        rf_confidence = rf.get(
            "confidence"
        )

        if rf_class:

            if rf_confidence is not None:

                findings.append(
                    f"Random Forest predicted "
                    f"{rf_class} with "
                    f"{rf_confidence:.2%} confidence."
                )

            else:

                findings.append(
                    f"Random Forest predicted "
                    f"{rf_class}."
                )

        # Two-Stream
        two_stream_class = two_stream.get(
            "multiclass_prediction"
        )

        two_stream_confidence = two_stream.get(
            "multiclass_confidence"
        )

        if two_stream_class:

            if two_stream_confidence is not None:

                findings.append(
                    f"Two-Stream model predicted "
                    f"{two_stream_class} with "
                    f"{two_stream_confidence:.2%} confidence."
                )

            else:

                findings.append(
                    f"Two-Stream model predicted "
                    f"{two_stream_class}."
                )

        # Model agreement
        if consistency.get("model_disagreement"):

            findings.append(
                "The supervised classifiers disagree "
                "on the specific threat category."
            )

        else:

            findings.append(
                "The supervised classifiers agree "
                "on the predicted threat category."
            )

        # Anomaly detection
        if anomaly.get("is_anomaly"):

            findings.append(
                "Isolation Forest detected a deviation "
                "from the learned normal traffic distribution."
            )

        else:

            findings.append(
                "Isolation Forest did not flag the "
                "sample as anomalous."
            )

        # Evidence strength
        strength_level = evidence_strength.get(
            "level"
        )

        strength_score = evidence_strength.get(
            "score"
        )

        if strength_level:

            if strength_score is not None:

                findings.append(
                    f"Overall evidence strength is "
                    f"{strength_level} "
                    f"({strength_score}/100)."
                )

            else:

                findings.append(
                    f"Overall evidence strength is "
                    f"{strength_level}."
                )

        # Threat intelligence
        mappings = threat_intelligence.get(
            "mitre_mappings",
            threat_intelligence.get(
                "mappings",
                [],
            ),
        )

        if mappings:

            findings.append(
                f"Threat intelligence provided "
                f"{len(mappings)} candidate MITRE ATT&CK "
                f"mapping(s) for analyst review."
            )

        return findings

    # ------------------------------------------------------------------
    # Explainability summary
    # ------------------------------------------------------------------

    def _summarize_explainability(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:

        explainability = evidence.get(
            "explainability",
            {},
        )

        shap_data = explainability.get(
            "shap",
            explainability.get(
                "local_shap",
                {},
            ),
        )

        lime_data = explainability.get(
            "lime",
            explainability.get(
                "local_lime",
                {},
            ),
        )

        return {
            "shap": shap_data,
            "lime": lime_data,
            "interpretation": (
                "SHAP and LIME provide local model "
                "explanations. They describe feature "
                "contributions to the prediction and "
                "should not be interpreted as causal evidence."
            ),
        }

    # ------------------------------------------------------------------
    # MITRE information
    # ------------------------------------------------------------------

    def _extract_mitre_information(
        self,
        threat_intelligence: Dict[str, Any],
    ) -> Dict[str, Any]:

        mappings = threat_intelligence.get(
            "mitre_mappings",
            threat_intelligence.get(
                "mappings",
                [],
            ),
        )

        guidance = threat_intelligence.get(
            "analyst_guidance",
            [],
        )

        policy = threat_intelligence.get(
            "mapping_policy",
            (
                "ATT&CK mappings are candidate mappings "
                "based on flow-level evidence and require "
                "additional telemetry for confirmation."
            ),
        )

        return {
            "mappings": mappings,
            "analyst_guidance": guidance,
            "mapping_policy": policy,
        }

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> List[str]:

        reasoning = []

        binary = evidence.get(
            "binary_detection",
            {},
        )

        classification = evidence.get(
            "attack_classification",
            {},
        )

        rf = classification.get(
            "random_forest",
            {},
        )

        two_stream = classification.get(
            "two_stream",
            {},
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {},
        )

        consistency = evidence.get(
            "model_consistency",
            {},
        )

        evidence_strength = evidence.get(
            "evidence_strength",
            {},
        )

        # Binary evidence
        if binary.get("prediction") == "Attack":

            reasoning.append(
                "The binary detector provides evidence "
                "that the traffic may represent malicious activity."
            )

        else:

            reasoning.append(
                "The binary detector does not provide "
                "strong evidence of malicious activity."
            )

        # Classifier comparison
        rf_class = rf.get(
            "predicted_class"
        )

        two_stream_class = two_stream.get(
            "multiclass_prediction"
        )

        if (
            rf_class
            and two_stream_class
            and rf_class == two_stream_class
        ):

            reasoning.append(
                "Random Forest and the Two-Stream classifier "
                "produce the same threat category, increasing "
                "classification consistency."
            )

        elif rf_class or two_stream_class:

            reasoning.append(
                "The supervised classifiers produce different "
                "threat categories, reducing confidence in "
                "the exact attack classification."
            )

        # Anomaly evidence
        if anomaly.get("is_anomaly"):

            reasoning.append(
                "Isolation Forest independently identifies "
                "the traffic as deviating from learned normal behavior."
            )

        else:

            reasoning.append(
                "Isolation Forest does not independently "
                "support the presence of an anomaly."
            )

        # Evidence strength
        strength_level = evidence_strength.get(
            "level"
        )

        if strength_level == "Low":

            reasoning.append(
                "The overall evidence strength is low, so "
                "the investigation result should be treated "
                "as preliminary."
            )

        elif strength_level == "Medium":

            reasoning.append(
                "The available evidence provides moderate "
                "support for the investigation assessment."
            )

        elif strength_level == "High":

            reasoning.append(
                "Multiple evidence sources provide strong "
                "support for the investigation assessment."
            )

        # Threat intelligence
        mappings = threat_intelligence.get(
            "mitre_mappings",
            threat_intelligence.get(
                "mappings",
                [],
            ),
        )

        if mappings:

            reasoning.append(
                "Threat intelligence provides candidate "
                "MITRE ATT&CK techniques that can guide "
                "additional investigation."
            )

        return reasoning

    # ------------------------------------------------------------------
    # Agent confidence
    # ------------------------------------------------------------------

    def _calculate_agent_confidence(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> Dict[str, Any]:

        binary = evidence.get(
            "binary_detection",
            {},
        )

        classification = evidence.get(
            "attack_classification",
            {},
        )

        rf = classification.get(
            "random_forest",
            {},
        )

        two_stream = classification.get(
            "two_stream",
            {},
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {},
        )

        consistency = evidence.get(
            "model_consistency",
            {},
        )

        evidence_strength = evidence.get(
            "evidence_strength",
            {},
        )

        score = 0.0

        # Binary confidence
        attack_probability = binary.get(
            "attack_probability"
        )

        if attack_probability is not None:

            binary_strength = abs(
                attack_probability - 0.5
            ) * 2

            score += binary_strength * 25

        # RF confidence
        rf_confidence = rf.get(
            "confidence"
        )

        if rf_confidence is not None:

            score += rf_confidence * 20

        # Two-Stream confidence
        two_stream_confidence = two_stream.get(
            "multiclass_confidence"
        )

        if two_stream_confidence is not None:

            score += two_stream_confidence * 20

        # Model agreement
        if not consistency.get(
            "model_disagreement",
            False,
        ):

            score += 15

        # Anomaly support
        if anomaly.get(
            "is_anomaly",
            False,
        ):

            score += 10

        # Evidence strength
        evidence_score = evidence_strength.get(
            "score"
        )

        if evidence_score is not None:

            score += (
                min(
                    max(
                        evidence_score,
                        0,
                    ),
                    100,
                )
                / 100
                * 10
            )

        score = min(
            max(score, 0.0),
            100.0,
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
            "score": round(
                score,
                2,
            ),
            "level": level,
            "interpretation": (
                "Agent confidence represents the "
                "strength and consistency of the "
                "available evidence. It is not an "
                "attack probability."
            ),
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> List[str]:

        recommendations = []

        threat_class = self._get_threat_class(
            evidence,
            threat_profile,
        )

        severity = self._get_severity(
            threat_profile,
        )

        severity_level = severity.get(
            "level",
            "Unknown",
        )

        anomaly = evidence.get(
            "anomaly_detection",
            {},
        )

        consistency = evidence.get(
            "model_consistency",
            {},
        )

        # Severity-based recommendation
        if severity_level == "Critical":

            recommendations.append(
                "Prioritize immediate analyst investigation "
                "and containment review."
            )

        elif severity_level == "High":

            recommendations.append(
                "Prioritize the event for analyst investigation "
                "and review related network activity."
            )

        elif severity_level == "Medium":

            recommendations.append(
                "Review the event and collect additional "
                "evidence before escalation."
            )

        elif severity_level == "Low":

            recommendations.append(
                "Monitor the event and correlate it with "
                "additional telemetry."
            )

        else:

            recommendations.append(
                "Treat the event as informational unless "
                "additional evidence increases its priority."
            )

        # Threat-class recommendation
        if threat_class == "DoS":

            recommendations.append(
                "Review traffic volume, affected services, "
                "and repeated connection patterns."
            )

        elif threat_class == "Reconnaissance":

            recommendations.append(
                "Review scanning behavior and contacted "
                "network services."
            )

        elif threat_class == "Exploits":

            recommendations.append(
                "Check affected services and correlate with "
                "host or application logs for exploitation evidence."
            )

        elif threat_class == "Backdoor":

            recommendations.append(
                "Review endpoint activity and command-and-control "
                "indicators for additional evidence."
            )

        elif threat_class == "Shellcode":

            recommendations.append(
                "Correlate the network event with endpoint "
                "execution and process telemetry."
            )

        elif threat_class == "Worms":

            recommendations.append(
                "Review lateral communication patterns and "
                "possible propagation between hosts."
            )

        elif threat_class == "Fuzzers":

            recommendations.append(
                "Review repeated malformed or unusual requests "
                "against exposed services."
            )

        # Anomaly recommendation
        if anomaly.get(
            "is_anomaly",
            False,
        ):

            recommendations.append(
                "Investigate the anomalous traffic against "
                "recent baseline behavior."
            )

        # Disagreement recommendation
        if consistency.get(
            "model_disagreement",
            False,
        ):

            recommendations.append(
                "Use additional telemetry because classifier "
                "disagreement reduces confidence in the exact "
                "threat category."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def _generate_verdict(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> str:

        threat_class = self._get_threat_class(
            evidence,
            threat_profile,
        )

        severity = self._get_severity(
            threat_profile,
        )

        severity_level = severity.get(
            "level",
            "Unknown",
        )

        binary = evidence.get(
            "binary_detection",
            {},
        )

        consistency = evidence.get(
            "model_consistency",
            {},
        )

        evidence_strength = evidence.get(
            "evidence_strength",
            {},
        )

        if (
            binary.get("prediction") == "Attack"
            and not consistency.get(
                "model_disagreement",
                False,
            )
            and evidence_strength.get(
                "level"
            ) in [
                "Medium",
                "High",
            ]
        ):

            return (
                f"Potential {threat_class} activity "
                f"requires investigation."
            )

        if (
            binary.get("prediction") == "Attack"
            and consistency.get(
                "model_disagreement",
                False,
            )
        ):

            return (
                "Potential malicious activity was detected, "
                "but the exact threat category is uncertain. "
                "Classifier disagreement reduces confidence "
                "in the exact attack category."
            )

        if (
            binary.get("prediction") == "Attack"
            and severity_level in [
                "Medium",
                "High",
                "Critical",
            ]
        ):

            return (
                f"Potential {threat_class} activity "
                f"requires further investigation."
            )

        if binary.get("prediction") == "Attack":

            return (
                "Potential malicious activity was detected, "
                "but additional evidence is required."
            )

        return (
            "No strong evidence of malicious activity was "
            "identified by the current detection pipeline."
        )

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------

    def _limitations(
        self,
    ) -> List[str]:

        return [
            (
                "The investigation agent operates on "
                "flow-level network features and does not "
                "have direct host, process, authentication, "
                "or packet-payload telemetry."
            ),
            (
                "Isolation Forest anomaly detection indicates "
                "deviation from learned normal traffic and "
                "does not by itself prove malicious activity."
            ),
            (
                "SHAP and LIME describe model behavior and "
                "feature contribution; they do not establish "
                "causal relationships."
            ),
            (
                "MITRE ATT&CK mappings are candidate mappings "
                "and require corroborating telemetry before "
                "being treated as confirmed techniques."
            ),
            (
                "Agent confidence represents evidence strength "
                "and consistency, not attack probability."
            ),
        ]

    # ------------------------------------------------------------------
    # Main investigation function
    # ------------------------------------------------------------------

    def investigate(
        self,
        evidence: Dict[str, Any],
        threat_profile: Dict[str, Any],
        threat_intelligence: Dict[str, Any],
    ) -> Dict[str, Any]:

        threat_class = self._get_threat_class(
            evidence,
            threat_profile,
        )

        severity = self._get_severity(
            threat_profile,
        )

        findings = self._generate_findings(
            evidence,
            threat_profile,
            threat_intelligence,
        )

        reasoning = self._generate_reasoning(
            evidence,
            threat_profile,
            threat_intelligence,
        )

        explainability = self._summarize_explainability(
            evidence,
        )

        mitre_information = self._extract_mitre_information(
            threat_intelligence,
        )

        # Phase 10: Transparent investigation confidence
        #
        # This replaces the previous manual confidence calculation
        # while preserving the existing evidence objects.
        confidence_result = self.confidence_engine.calculate(
            binary_detection=evidence.get(
                "binary_detection",
                {},
            ),
            classification=evidence.get(
                "attack_classification",
                {},
            ),
            anomaly_detection=evidence.get(
                "anomaly_detection",
                {},
            ),
            model_consistency=evidence.get(
                "model_consistency",
                {},
            ),
            evidence_strength=evidence.get(
                "evidence_strength",
                {},
            ),
            threat_intelligence=mitre_information,
            threat_class=threat_class,
        )

        recommendations = self._generate_recommendations(
            evidence,
            threat_profile,
            threat_intelligence,
        )

        verdict = self._generate_verdict(
            evidence,
            threat_profile,
            threat_intelligence,
        )

        limitations = self._limitations()

        result = {
            "agent": {
                "name": self.agent_name,
                "version": self.version,
                "type": "Evidence-grounded investigation agent",
            },

            "investigation_verdict": {
                "verdict": verdict,
                "threat_class": threat_class,
                "severity": severity,
                "confidence": confidence_result,
            },

            "threat_class": threat_class,

            "severity": severity,

            "agent_confidence": confidence_result,

            # Keep confidence as a direct compatibility field
            "confidence": confidence_result,

            # Required by the investigation runner
            "key_findings": findings,

            # Keep findings as a direct compatibility field
            "findings": findings,

            "reasoning": reasoning,

            "explainability": explainability,

            "mitre": mitre_information,

            "recommended_actions": recommendations,

            "limitations": limitations,
        }

        return result


# ----------------------------------------------------------------------
# Synthetic test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    agent = InvestigationAgent()

    synthetic_evidence = {

        "binary_detection": {
            "prediction": "Attack",
            "attack_probability": 0.95,
        },

        "attack_classification": {

            "random_forest": {
                "predicted_class": "Exploits",
                "confidence": 0.91,
            },

            "two_stream": {
                "multiclass_prediction": "Exploits",
                "multiclass_confidence": 0.88,
            },
        },

        "anomaly_detection": {
            "is_anomaly": True,
            "anomaly_score": 0.25,
        },

        "model_consistency": {
            "model_disagreement": False,
        },

        "evidence_strength": {
            "level": "High",
            "score": 90,
        },

        "explainability": {

            "shap": {
                "top_features": [
                    "dns_response_activity",
                    "protocol_distribution",
                ]
            },

            "lime": {
                "top_features": [
                    "flow_duration",
                    "query_rate",
                ]
            },
        },
    }

    synthetic_profile = {

        "threat_profile": {
            "predicted_threat": "Exploits",
        },

        "severity_assessment": {
            "severity_level": "High",
            "severity_score": 85.0,
        },
    }

    synthetic_intelligence = {

        "mitre_mappings": [
            {
                "technique_id": "T1190",
                "technique_name": (
                    "Exploit Public-Facing Application"
                ),
            }
        ],

        "analyst_guidance": [
            "Review affected services."
        ],

        "mapping_policy": (
            "ATT&CK mappings are candidate mappings "
            "based on flow-level evidence."
        ),
    }

    result = agent.investigate(
        evidence=synthetic_evidence,
        threat_profile=synthetic_profile,
        threat_intelligence=synthetic_intelligence,
    )

    print()

    print("Threat class:")
    print(result["threat_class"])

    print()

    print("Severity:")
    print(result["severity"])

    print()

    print("Agent confidence:")
    print(result["agent_confidence"])

    print()

    print("Verdict:")
    print(result["investigation_verdict"]["verdict"])

    print()

    print("Key findings:")
    print(len(result["key_findings"]))

    print()

    print("Confidence components:")

    for component, value in result[
        "agent_confidence"
    ].get(
        "components",
        {},
    ).items():

        print(
            f"{component}: {value}"
        )