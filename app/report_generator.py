"""
CyberAgent Incident Report Generator

Generates a structured Markdown incident report
from the existing CyberAgent investigation results.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "reports"
)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def safe_get(
    data: Dict[str, Any],
    *keys,
    default=None,
):
    """
    Safely retrieve nested dictionary values.
    """

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


# ----------------------------------------------------------------------
# Generate Markdown report
# ----------------------------------------------------------------------

def generate_incident_report(
    sample: Dict[str, Any],
) -> str:
    """
    Generate a Markdown incident report.
    """

    investigation = sample.get(
        "investigation",
        {},
    )

    evidence = sample.get(
        "evidence",
        {},
    )

    threat_profile = sample.get(
        "threat_profile",
        {},
    )

    threat_intelligence = sample.get(
        "threat_intelligence",
        {},
    )

    sample_number = sample.get(
        "sample_number",
        "Unknown",
    )

    # --------------------------------------------------------------
    # Investigation information
    # --------------------------------------------------------------

    investigation_verdict = investigation.get(
        "investigation_verdict",
        {},
    )

    verdict = investigation_verdict.get(
        "verdict",
        "Unknown",
    )

    threat_class = investigation.get(
        "threat_class",
        "Unknown",
    )

    severity = investigation.get(
        "severity",
        {},
    )

    severity_level = severity.get(
        "level",
        "Unknown",
    )

    severity_score = severity.get(
        "score",
        "N/A",
    )

    agent_confidence = investigation.get(
        "agent_confidence",
        investigation.get(
            "confidence",
            {},
        ),
    )

    confidence_score = agent_confidence.get(
        "score",
        "N/A",
    )

    confidence_level = agent_confidence.get(
        "level",
        "Unknown",
    )

    # --------------------------------------------------------------
    # Evidence information
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Threat intelligence
    # --------------------------------------------------------------

    intel_mappings = threat_intelligence.get(
        "mitre_mappings",
        threat_intelligence.get(
            "mappings",
            [],
        ),
    )

    # --------------------------------------------------------------
    # Findings
    # --------------------------------------------------------------

    findings = investigation.get(
        "key_findings",
        investigation.get(
            "findings",
            [],
        ),
    )

    # --------------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------------

    recommendations = investigation.get(
        "recommended_actions",
        [],
    )

    # --------------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------------

    reasoning = investigation.get(
        "reasoning",
        [],
    )

    # --------------------------------------------------------------
    # Build report
    # --------------------------------------------------------------

    lines = []

    lines.append(
        "# CyberAgent Incident Investigation Report"
    )

    lines.append("")

    lines.append(
        f"**Sample:** {sample_number}"
    )

    lines.append(
        f"**Generated:** "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    lines.append("")

    lines.append("---")

    lines.append("")

    # --------------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------------

    lines.append(
        "## 1. Executive Summary"
    )

    lines.append("")

    lines.append(
        f"**Investigation Verdict:** {verdict}"
    )

    lines.append("")

    lines.append(
        f"**Threat Classification:** {threat_class}"
    )

    lines.append("")

    lines.append(
        f"**Severity:** "
        f"{severity_level} "
        f"({severity_score}/100)"
    )

    lines.append("")

    lines.append(
        f"**Agent Confidence:** "
        f"{confidence_level} "
        f"({confidence_score}/100)"
    )

    lines.append("")

    # --------------------------------------------------------------
    # Detection
    # --------------------------------------------------------------

    lines.append(
        "## 2. Detection Assessment"
    )

    lines.append("")

    binary_prediction = binary.get(
        "prediction",
        "Unknown",
    )

    attack_probability = binary.get(
        "attack_probability",
    )

    lines.append(
        f"- Binary prediction: **{binary_prediction}**"
    )

    if attack_probability is not None:

        lines.append(
            f"- Attack probability: "
            f"**{attack_probability:.2%}**"
        )

    lines.append(
        f"- Random Forest prediction: "
        f"**{rf.get('predicted_class', 'Unknown')}**"
    )

    if rf.get("confidence") is not None:

        lines.append(
            f"- Random Forest confidence: "
            f"**{rf['confidence']:.2%}**"
        )

    lines.append(
        f"- Two-Stream prediction: "
        f"**{two_stream.get('multiclass_prediction', 'Unknown')}**"
    )

    if two_stream.get(
        "multiclass_confidence"
    ) is not None:

        lines.append(
            f"- Two-Stream confidence: "
            f"**{two_stream['multiclass_confidence']:.2%}**"
        )

    lines.append("")

    # --------------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------------

    lines.append(
        "## 3. Anomaly Assessment"
    )

    lines.append("")

    anomaly_flag = anomaly.get(
        "is_anomaly",
        False,
    )

    lines.append(
        f"- Isolation Forest anomaly: "
        f"**{'Yes' if anomaly_flag else 'No'}**"
    )

    if anomaly.get("anomaly_score") is not None:

        lines.append(
            f"- Anomaly score: "
            f"**{anomaly['anomaly_score']:.6f}**"
        )

    lines.append("")

    # --------------------------------------------------------------
    # Model consistency
    # --------------------------------------------------------------

    lines.append(
        "## 4. Model Consistency"
    )

    lines.append("")

    disagreement = consistency.get(
        "model_disagreement",
        False,
    )

    lines.append(
        f"- Classifier disagreement: "
        f"**{'Yes' if disagreement else 'No'}**"
    )

    strength_level = evidence_strength.get(
        "level",
        "Unknown",
    )

    strength_score = evidence_strength.get(
        "score",
        "N/A",
    )

    lines.append(
        f"- Evidence strength: "
        f"**{strength_level} ({strength_score}/100)**"
    )

    lines.append("")

    # --------------------------------------------------------------
    # Key findings
    # --------------------------------------------------------------

    lines.append(
        "## 5. Key Findings"
    )

    lines.append("")

    if findings:

        for finding in findings:

            lines.append(
                f"- {finding}"
            )

    else:

        lines.append(
            "- No structured findings available."
        )

    lines.append("")

    # --------------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------------

    lines.append(
        "## 6. Investigation Reasoning"
    )

    lines.append("")

    if reasoning:

        for item in reasoning:

            lines.append(
                f"- {item}"
            )

    else:

        lines.append(
            "- No reasoning information available."
        )

    lines.append("")

    # --------------------------------------------------------------
    # MITRE ATT&CK
    # --------------------------------------------------------------

    lines.append(
        "## 7. MITRE ATT&CK Candidate Mappings"
    )

    lines.append("")

    if intel_mappings:

        for mapping in intel_mappings:

            technique_id = mapping.get(
                "technique_id",
                "N/A",
            )

            technique_name = mapping.get(
                "technique_name",
                "Unknown",
            )

            tactic = mapping.get(
                "tactic",
                "Unknown",
            )

            relevance = mapping.get(
                "relevance",
                "Unknown",
            )

            confidence = mapping.get(
                "confidence",
                "Unknown",
            )

            lines.append(
                f"- **{technique_id} — "
                f"{technique_name}**"
            )

            lines.append(
                f"  - Tactic: {tactic}"
            )

            lines.append(
                f"  - Relevance: {relevance}"
            )

            lines.append(
                f"  - Mapping confidence: {confidence}"
            )

    else:

        lines.append(
            "- No candidate MITRE ATT&CK mappings."
        )

    lines.append("")

    lines.append(
        "> ATT&CK mappings are candidate mappings based "
        "on flow-level evidence and require corroboration "
        "with additional telemetry before being treated "
        "as confirmed techniques."
    )

    lines.append("")

    # --------------------------------------------------------------
    # Recommended actions
    # --------------------------------------------------------------

    lines.append(
        "## 8. Recommended Actions"
    )

    lines.append("")

    if recommendations:

        for recommendation in recommendations:

            lines.append(
                f"- {recommendation}"
            )

    else:

        lines.append(
            "- No additional recommendations available."
        )

    lines.append("")

    # --------------------------------------------------------------
    # Limitations
    # --------------------------------------------------------------

    limitations = investigation.get(
        "limitations",
        [],
    )

    lines.append(
        "## 9. Investigation Limitations"
    )

    lines.append("")

    if limitations:

        for limitation in limitations:

            lines.append(
                f"- {limitation}"
            )

    else:

        lines.append(
            "- No limitations provided."
        )

    lines.append("")

    lines.append("---")

    lines.append("")

    lines.append(
        "**CyberAgent:** Agent-Assisted Network Threat "
        "Detection and Investigation using Machine Learning"
    )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Save report
# ----------------------------------------------------------------------

def save_incident_report(
    sample: Dict[str, Any],
) -> Path:
    """
    Generate and save an incident report.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_number = sample.get(
        "sample_number",
        "unknown",
    )

    report = generate_incident_report(
        sample
    )

    report_path = (
        REPORT_DIR
        / f"incident_report_sample_{sample_number}.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    return report_path


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    from data_loader import (
        build_sample_dataset,
    )

    print("=" * 70)
    print("CYBERAGENT INCIDENT REPORT GENERATOR")
    print("=" * 70)

    samples = build_sample_dataset()

    if not samples:

        print()
        print("No samples available.")
        raise SystemExit(1)

    report_path = save_incident_report(
        samples[0]
    )

    print()
    print(
        f"Report generated successfully:"
    )

    print(report_path)