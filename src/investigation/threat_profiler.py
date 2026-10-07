"""
CyberAgent Threat Profiler

Converts ML investigation evidence into a structured
threat profile.

This module does NOT retrain any ML model.

It interprets:
- Binary detection
- Random Forest classification
- Two-Stream classification
- Anomaly detection
- Model agreement
- Evidence strength

The output is an analyst-friendly threat profile.
"""


class ThreatProfiler:

    THREAT_PROFILES = {

        "Normal": {
            "category": "Benign Traffic",
            "description": (
                "Traffic classified as normal or non-malicious "
                "by the multiclass classifier."
            ),
            "behavior": "Normal network behavior",
            "priority": "Low",
            "investigation_focus": [
                "Verify whether the traffic is expected",
                "Check for disagreement between binary and multiclass models"
            ]
        },

        "Generic": {
            "category": "Generic Attack",
            "description": (
                "A broad attack pattern that does not correspond "
                "to a more specific attack category."
            ),
            "behavior": "Generic malicious network activity",
            "priority": "Medium",
            "investigation_focus": [
                "Review source and destination behavior",
                "Check repeated or unusual connection patterns",
                "Compare with historical traffic"
            ]
        },

        "Exploits": {
            "category": "Exploitation",
            "description": (
                "Traffic associated with attempts to exploit "
                "vulnerabilities or abnormal application behavior."
            ),
            "behavior": "Possible vulnerability exploitation",
            "priority": "High",
            "investigation_focus": [
                "Identify targeted service",
                "Check for repeated exploitation attempts",
                "Review affected host activity"
            ]
        },

        "Fuzzers": {
            "category": "Fuzzing",
            "description": (
                "Traffic associated with abnormal or malformed "
                "input patterns used to probe a service."
            ),
            "behavior": "Input or protocol fuzzing",
            "priority": "High",
            "investigation_focus": [
                "Inspect unusual request patterns",
                "Check targeted services",
                "Look for repeated malformed traffic"
            ]
        },

        "DoS": {
            "category": "Denial of Service",
            "description": (
                "Traffic associated with attempts to overwhelm "
                "or disrupt a network service."
            ),
            "behavior": "Service disruption or traffic flooding",
            "priority": "Critical",
            "investigation_focus": [
                "Check traffic volume",
                "Identify targeted host or service",
                "Look for sustained or repeated traffic bursts"
            ]
        },

        "Reconnaissance": {
            "category": "Reconnaissance",
            "description": (
                "Traffic associated with information gathering, "
                "scanning, or probing of network services."
            ),
            "behavior": "Network or service discovery",
            "priority": "Medium",
            "investigation_focus": [
                "Identify scanned services",
                "Check connection diversity",
                "Look for repeated probing activity"
            ]
        },

        "Analysis": {
            "category": "Analysis Activity",
            "description": (
                "Traffic classified as analysis-related activity "
                "within the UNSW-NB15 attack taxonomy."
            ),
            "behavior": "Suspicious analysis-related traffic",
            "priority": "Medium",
            "investigation_focus": [
                "Inspect communication pattern",
                "Compare with known traffic behavior",
                "Check for associated suspicious flows"
            ]
        },

        "Backdoor": {
            "category": "Backdoor",
            "description": (
                "Traffic potentially associated with unauthorized "
                "remote access or persistent communication."
            ),
            "behavior": "Potential unauthorized remote access",
            "priority": "Critical",
            "investigation_focus": [
                "Identify communicating hosts",
                "Check unusual outbound connections",
                "Review persistence-related activity"
            ]
        },

        "Shellcode": {
            "category": "Shellcode",
            "description": (
                "Traffic associated with patterns classified "
                "as shellcode-related activity."
            ),
            "behavior": "Potential payload execution activity",
            "priority": "Critical",
            "investigation_focus": [
                "Identify targeted host",
                "Review suspicious payload-related traffic",
                "Check for related exploitation activity"
            ]
        },

        "Worms": {
            "category": "Worm Propagation",
            "description": (
                "Traffic potentially associated with automated "
                "propagation or self-spreading behavior."
            ),
            "behavior": "Potential automated propagation",
            "priority": "Critical",
            "investigation_focus": [
                "Identify multiple contacted hosts",
                "Check repeated connection patterns",
                "Look for signs of propagation"
            ]
        }
    }

    def __init__(self):
        pass

    def _get_predicted_class(self, evidence):
        """
        Determine the most useful predicted class.

        We prefer the Random Forest prediction when available
        because Random Forest is the selected multiclass model.
        """

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

        rf_class = rf.get("predicted_class")

        if rf_class:
            return rf_class

        return two_stream.get(
            "multiclass_prediction",
            "Unknown"
        )

    def generate_profile(self, evidence):

        predicted_class = self._get_predicted_class(
            evidence
        )

        profile = self.THREAT_PROFILES.get(
            predicted_class,
            {
                "category": "Unknown Threat",
                "description": (
                    "The traffic does not match a known "
                    "threat profile."
                ),
                "behavior": "Unknown",
                "priority": "Review",
                "investigation_focus": [
                    "Review model predictions",
                    "Inspect explainability results",
                    "Perform additional investigation"
                ]
            }
        )

        binary_detection = evidence.get(
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

        evidence_strength = evidence.get(
            "evidence_strength",
            {}
        )

        return {
            "predicted_threat": predicted_class,

            "threat_category": profile["category"],

            "threat_description": profile[
                "description"
            ],

            "behavior": profile["behavior"],

            "base_priority": profile["priority"],

            "binary_detection": {
                "prediction": binary_detection.get(
                    "prediction"
                ),
                "attack_probability": binary_detection.get(
                    "attack_probability"
                )
            },

            "classifier_assessment": {
                "random_forest_class": rf.get(
                    "predicted_class"
                ),
                "random_forest_confidence": rf.get(
                    "confidence"
                ),
                "two_stream_class": two_stream.get(
                    "multiclass_prediction"
                ),
                "two_stream_confidence": two_stream.get(
                    "multiclass_confidence"
                )
            },

            "anomaly_assessment": {
                "is_anomaly": anomaly.get(
                    "is_anomaly"
                ),
                "anomaly_score": anomaly.get(
                    "anomaly_score"
                )
            },

            "model_consistency": {
                "status": consistency.get(
                    "agreement_status"
                ),
                "model_disagreement": consistency.get(
                    "model_disagreement"
                )
            },

            "evidence_strength": {
                "score": evidence_strength.get(
                    "score"
                ),
                "level": evidence_strength.get(
                    "level"
                )
            },

            "investigation_focus": profile[
                "investigation_focus"
            ]
        }