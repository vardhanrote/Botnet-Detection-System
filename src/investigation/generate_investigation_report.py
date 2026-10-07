"""
Generate a human-readable investigation report
from the CyberAgent Investigation Agent output.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "investigation_agent_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "investigation_summary.txt"
)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    results = data.get(
        "results",
        []
    )

    lines = []

    lines.append("=" * 70)
    lines.append("CYBERAGENT INVESTIGATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"Total investigations: {len(results)}"
    )

    lines.append("")

    for number, result in enumerate(
        results,
        start=1,
    ):

        verdict = result.get(
            "investigation_verdict",
            {}
        )

        confidence = result.get(
            "agent_confidence",
            {}
        )

        lines.append("-" * 70)
        lines.append(
            f"INVESTIGATION {number}"
        )
        lines.append("-" * 70)

        if "sample_index" in result:
            lines.append(
                f"Sample: {result['sample_index']}"
            )

        lines.append(
            f"Threat class: "
            f"{result.get('threat_class', 'Unknown')}"
        )

        lines.append(
            f"Severity: "
            f"{verdict.get('severity', 'Unknown')}"
        )

        lines.append(
            f"Agent confidence: "
            f"{confidence.get('level', 'Unknown')} "
            f"({confidence.get('score', 0):.2f}/100)"
        )

        lines.append("")

        lines.append("FINAL VERDICT:")
        lines.append(
            verdict.get(
                "verdict",
                "No verdict available."
            )
        )

        lines.append("")

        lines.append("KEY FINDINGS:")

        for finding in result.get(
            "key_findings",
            [],
        ):
            lines.append(
                f"- {finding}"
            )

        lines.append("")

        lines.append(
            "RECOMMENDED ANALYST ACTIONS:"
        )

        for action in result.get(
            "recommended_actions",
            [],
        ):
            lines.append(
                f"- {action}"
            )

        lines.append("")

        mitre = result.get(
            "mitre_attack",
            {}
        )

        mappings = mitre.get(
            "candidate_techniques",
            []
        )

        lines.append(
            "MITRE ATT&CK CANDIDATE MAPPINGS:"
        )

        if mappings:

            for mapping in mappings:

                technique_id = mapping.get(
                    "technique_id",
                    "N/A"
                )

                technique_name = mapping.get(
                    "technique_name",
                    "Unknown"
                )

                lines.append(
                    f"- {technique_id}: "
                    f"{technique_name}"
                )

        else:
            lines.append(
                "- No candidate technique mapping."
            )

        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(
        f"Investigation report saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()