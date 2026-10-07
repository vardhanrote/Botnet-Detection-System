"""
Run the CyberAgent Investigation Agent.

Inputs:
- evidence_extraction_results.json
- threat_profiling_results.json
- threat_intelligence_results.json

Output:
- investigation_agent_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

from src.investigation.investigation_agent import (
    InvestigationAgent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "evidence_extraction_results.json"
)

THREAT_PROFILE_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "threat_profiling_results.json"
)

THREAT_INTEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "threat_intelligence"
    / "threat_intelligence_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "investigation_agent_results.json"
)


def load_json(path: Path):
    """
    Load JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():

    print("=" * 70)
    print("CYBERAGENT AI INVESTIGATION AGENT")
    print("=" * 70)

    print("\nLoading evidence...")

    evidence_data = load_json(
        EVIDENCE_FILE
    )

    threat_profile_data = load_json(
        THREAT_PROFILE_FILE
    )

    threat_intel_data = load_json(
        THREAT_INTEL_FILE
    )

    print("Evidence loaded.")
    print("Threat profiles loaded.")
    print("Threat intelligence loaded.")

    # --------------------------------------------------------------
    # Normalize profile records
    # --------------------------------------------------------------

    evidence_records = (
        evidence_data
        if isinstance(evidence_data, list)
        else evidence_data.get(
            "results",
            evidence_data.get(
                "samples",
                [],
            ),
        )
    )

    profile_records = (
        threat_profile_data
        if isinstance(threat_profile_data, list)
        else threat_profile_data.get(
            "results",
            threat_profile_data.get(
                "profiles",
                [],
            ),
        )
    )

    intel_records = (
        threat_intel_data
        if isinstance(threat_intel_data, list)
        else threat_intel_data.get(
            "results",
            threat_intel_data.get(
                "profiles",
                [],
            ),
        )
    )

    if not isinstance(
        evidence_records,
        list,
    ):
        raise ValueError(
            "Could not find evidence records."
        )

    agent = InvestigationAgent()

    results = []

    print(
        f"\nInvestigating {len(evidence_records)} samples..."
    )

    for index, evidence in enumerate(
        evidence_records
    ):

        # ----------------------------------------------------------
        # Match corresponding profile
        # ----------------------------------------------------------

        profile = {}

        if index < len(profile_records):
            profile = profile_records[index]

        intel = {}

        if index < len(intel_records):
            intel = intel_records[index]

        # ----------------------------------------------------------
        # Run agent
        # ----------------------------------------------------------

        result = agent.investigate(
            evidence=evidence,
            threat_profile=profile,
            threat_intelligence=intel,
        )

        # Preserve sample identifier where available
        if isinstance(evidence, dict):

            sample_id = (
                evidence.get("sample_index")
                or evidence.get("index")
                or evidence.get("sample_id")
            )

            if sample_id is not None:
                result["sample_index"] = sample_id

        results.append(result)

        # ----------------------------------------------------------
        # Console output
        # ----------------------------------------------------------

        print("\n" + "-" * 70)

        if "sample_index" in result:
            print(
                f"Sample               : "
                f"{result['sample_index']}"
            )

        print(
            f"Threat class         : "
            f"{result['threat_class']}"
        )

        verdict = result[
            "investigation_verdict"
        ]

        print(
            f"Verdict              : "
            f"{verdict['verdict']}"
        )

        print(
            f"Severity             : "
            f"{verdict['severity']}"
        )

        confidence = result[
            "agent_confidence"
        ]

        print(
            f"Agent confidence     : "
            f"{confidence['level']} "
            f"({confidence['score']:.2f}/100)"
        )

        components = confidence.get(
            "components",
            {}
        )

        print("Confidence components:")

        print(
            f"  Binary detection    : "
            f"{components.get('binary_detection', 0):.2f}"
        )

        print(
            f"  Classifier confidence: "
            f"{components.get('classifier_confidence', 0):.2f}"
        )

        print(
            f"  Model agreement     : "
            f"{components.get('model_agreement', 0):.2f}"
        )

        print(
            f"  Evidence strength   : "
            f"{components.get('evidence_strength', 0):.2f}"
        )

        print(
            f"  Anomaly evidence    : "
            f"{components.get('anomaly_evidence', 0):.2f}"
        )

        print(
            f"  Threat intelligence: "
            f"{components.get('threat_intelligence_support', 0):.2f}"
        )

        print(
            "Key findings         : "
            f"{len(result['key_findings'])}"
        )

        print(
            "Recommended actions  : "
            f"{len(result['recommended_actions'])}"
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "agent": {
            "name": agent.agent_name,
            "version": agent.version,
        },
        "investigation_count": len(results),
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved results to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()