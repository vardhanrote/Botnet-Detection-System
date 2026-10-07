"""
CyberAgent Local Threat Intelligence Knowledge Base

This file contains a small, deterministic knowledge base
for mapping UNSW-NB15 attack classes to candidate
MITRE ATT&CK techniques.

IMPORTANT:
These mappings are candidate mappings.

A dataset attack label alone does NOT prove that an
ATT&CK technique occurred.

The system therefore stores:
- Technique ID
- Technique name
- MITRE tactic
- Relevance
- Required corroborating evidence
- Analyst guidance
"""


MITRE_KNOWLEDGE_BASE = {

    "Normal": {

        "description": (
            "Traffic classified as normal by the "
            "multiclass classifier."
        ),

        "techniques": []
    },


    "Generic": {

        "description": (
            "Broad malicious traffic that does not "
            "map cleanly to a specific ATT&CK technique "
            "using flow-level evidence alone."
        ),

        "techniques": [

            {
                "technique_id": None,

                "technique_name": (
                    "No direct ATT&CK mapping"
                ),

                "tactic": "Unknown",

                "relevance": "Low",

                "confidence": "Low",

                "reason": (
                    "The Generic class is broad and "
                    "does not provide enough behavioral "
                    "evidence for a specific ATT&CK technique."
                ),

                "required_evidence": [
                    "Source and destination context",
                    "Protocol/service information",
                    "Repeated traffic behavior",
                    "Host-level telemetry"
                ]
            }
        ]
    },


    "Exploits": {

        "description": (
            "Traffic associated with exploitation-related "
            "network behavior."
        ),

        "techniques": [

            {
                "technique_id": "T1190",

                "technique_name": (
                    "Exploit Public-Facing Application"
                ),

                "tactic": "Initial Access",

                "relevance": "High",

                "confidence": "Medium",

                "reason": (
                    "Exploitation traffic may be consistent "
                    "with attempts to exploit an exposed "
                    "application or network service."
                ),

                "required_evidence": [
                    "Internet-facing target",
                    "Exploit-like request pattern",
                    "Target service information",
                    "Application or server logs",
                    "Evidence of exploitation outcome"
                ]
            },

            {
                "technique_id": "T1210",

                "technique_name": (
                    "Exploitation of Remote Services"
                ),

                "tactic": "Lateral Movement",

                "relevance": "Medium",

                "confidence": "Low",

                "reason": (
                    "Exploitation-related traffic can also "
                    "be relevant to remote-service exploitation, "
                    "but network-flow evidence alone cannot "
                    "establish lateral movement."
                ),

                "required_evidence": [
                    "Internal source and destination",
                    "Remote service identification",
                    "Vulnerability evidence",
                    "Successful access or execution evidence"
                ]
            }
        ]
    },


    "Fuzzers": {

        "description": (
            "Traffic associated with fuzzing or malformed "
            "input behavior."
        ),

        "techniques": [

            {
                "technique_id": "T1190",

                "technique_name": (
                    "Exploit Public-Facing Application"
                ),

                "tactic": "Initial Access",

                "relevance": "Medium",

                "confidence": "Low",

                "reason": (
                    "Fuzzing can be used to discover or exploit "
                    "weaknesses in exposed services, but fuzzing "
                    "alone does not prove exploitation."
                ),

                "required_evidence": [
                    "Repeated malformed requests",
                    "Targeted exposed service",
                    "Application error behavior",
                    "Evidence of successful exploitation"
                ]
            }
        ]
    },


    "DoS": {

        "description": (
            "Traffic associated with denial-of-service "
            "behavior."
        ),

        "techniques": [

            {
                "technique_id": "T1499",

                "technique_name": (
                    "Endpoint Denial of Service"
                ),

                "tactic": "Impact",

                "relevance": "High",

                "confidence": "Medium",

                "reason": (
                    "High-volume or disruptive traffic can "
                    "be consistent with attempts to degrade "
                    "or block service availability."
                ),

                "required_evidence": [
                    "Traffic volume increase",
                    "Targeted service",
                    "Resource exhaustion",
                    "Service degradation or outage",
                    "Sustained attack pattern"
                ]
            }
        ]
    },


    "Reconnaissance": {

        "description": (
            "Traffic associated with network probing, "
            "scanning, or information gathering."
        ),

        "techniques": [

            {
                "technique_id": "T1046",

                "technique_name": (
                    "Network Service Discovery"
                ),

                "tactic": "Discovery",

                "relevance": "High",

                "confidence": "Medium",

                "reason": (
                    "Reconnaissance traffic may correspond "
                    "to attempts to identify services or "
                    "network endpoints."
                ),

                "required_evidence": [
                    "Multiple destination hosts or ports",
                    "Sequential probing",
                    "Port scanning behavior",
                    "Service enumeration",
                    "Time-windowed network activity"
                ]
            }
        ]
    },


    "Analysis": {

        "description": (
            "Traffic classified as analysis-related "
            "activity in UNSW-NB15."
        ),

        "techniques": [

            {
                "technique_id": None,

                "technique_name": (
                    "No direct ATT&CK mapping"
                ),

                "tactic": "Unknown",

                "relevance": "Low",

                "confidence": "Low",

                "reason": (
                    "The Analysis class does not contain "
                    "enough semantic information to confidently "
                    "select a single ATT&CK technique."
                ),

                "required_evidence": [
                    "Application context",
                    "Host telemetry",
                    "Process activity",
                    "Communication context"
                ]
            }
        ]
    },


    "Backdoor": {

        "description": (
            "Traffic potentially associated with "
            "unauthorized remote access or persistent "
            "communication."
        ),

        "techniques": [

            {
                "technique_id": None,

                "technique_name": (
                    "Candidate technique requires "
                    "host-level corroboration"
                ),

                "tactic": "Command and Control",

                "relevance": "Medium",

                "confidence": "Low",

                "reason": (
                    "A backdoor classification suggests "
                    "suspicious unauthorized communication, "
                    "but the available flow features do not "
                    "identify the exact ATT&CK technique."
                ),

                "required_evidence": [
                    "Persistent communication pattern",
                    "Destination reputation",
                    "Host process telemetry",
                    "Persistence mechanism",
                    "Command-and-control indicators"
                ]
            }
        ]
    },


    "Shellcode": {

        "description": (
            "Traffic associated with shellcode-related "
            "activity in the dataset."
        ),

        "techniques": [

            {
                "technique_id": None,

                "technique_name": (
                    "Candidate exploitation technique "
                    "requires corroboration"
                ),

                "tactic": "Execution",

                "relevance": "Medium",

                "confidence": "Low",

                "reason": (
                    "Shellcode-related classification may "
                    "indicate exploitation or payload execution, "
                    "but flow-level features cannot prove the "
                    "specific ATT&CK execution technique."
                ),

                "required_evidence": [
                    "Exploit evidence",
                    "Payload indicators",
                    "Process creation telemetry",
                    "Memory or endpoint telemetry",
                    "Successful execution evidence"
                ]
            }
        ]
    },


    "Worms": {

        "description": (
            "Traffic potentially associated with automated "
            "propagation across systems."
        ),

        "techniques": [

            {
                "technique_id": "T1210",

                "technique_name": (
                    "Exploitation of Remote Services"
                ),

                "tactic": "Lateral Movement",

                "relevance": "Medium",

                "confidence": "Low",

                "reason": (
                    "Automated propagation may involve "
                    "exploitation of remote services, but "
                    "the dataset flow alone cannot establish "
                    "the propagation mechanism."
                ),

                "required_evidence": [
                    "Multiple target hosts",
                    "Repeated connection attempts",
                    "Remote service identification",
                    "Evidence of successful exploitation",
                    "Propagation across systems"
                ]
            }
        ]
    }
}


def get_threat_intelligence(attack_class):

    """
    Return the knowledge-base entry for an attack class.

    Unknown classes receive a safe fallback result.
    """

    return MITRE_KNOWLEDGE_BASE.get(
        attack_class,
        {
            "description": (
                "No threat intelligence profile "
                "is available for this class."
            ),
            "techniques": []
        }
    )