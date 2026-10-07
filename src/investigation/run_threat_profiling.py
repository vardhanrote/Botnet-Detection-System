"""
CyberAgent Threat Profiling Runner

Reads the existing evidence extraction results
and generates:

1. Threat profile
2. Severity score
3. Severity level
4. Analyst recommendations

No ML model is retrained.
"""

import json
from pathlib import Path

from src.investigation.threat_profiler import (
    ThreatProfiler
)

from src.investigation.severity_engine import (
    SeverityEngine
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "investigation"
    / "evidence_extraction_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "investigation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "threat_profiling_results.json"
)


def load_evidence():

    print("Loading evidence extraction results...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Evidence file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


def extract_samples(data):

    """
    Handles common JSON structures.

    The evidence extractor may store samples
    directly as a list or under a key such as
    'samples' or 'results'.
    """

    if isinstance(data, list):

        return data

    if isinstance(data, dict):

        if isinstance(
            data.get("samples"),
            list
        ):
            return data["samples"]

        if isinstance(
            data.get("results"),
            list
        ):
            return data["results"]

        if isinstance(
            data.get("investigations"),
            list
        ):
            return data["investigations"]

    raise ValueError(
        "Could not find investigation samples "
        "inside evidence extraction JSON."
    )


def main():

    print("=" * 70)
    print("CYBERAGENT THREAT PROFILING + SEVERITY ENGINE")
    print("=" * 70)

    data = load_evidence()

    samples = extract_samples(data)

    print(
        f"Evidence samples loaded: {len(samples)}"
    )

    profiler = ThreatProfiler()
    severity_engine = SeverityEngine()

    results = []

    for index, evidence in enumerate(samples):

        print()
        print("-" * 70)

        print(
            f"Analyzing investigation sample {index}"
        )

        threat_profile = (
            profiler.generate_profile(
                evidence
            )
        )

        severity = (
            severity_engine.evaluate(
                evidence
            )
        )

        combined_result = {

            "sample_number": (
                evidence.get(
                    "sample_index",
                    index
                )
            ),

            "threat_profile": threat_profile,

            "severity_assessment": severity
        }

        results.append(
            combined_result
        )

        print(
            "Threat class       : "
            f"{severity['threat_class']}"
        )

        print(
            "Threat category    : "
            f"{threat_profile['threat_category']}"
        )

        print(
            "Severity score     : "
            f"{severity['severity_score']}/100"
        )

        print(
            "Severity level     : "
            f"{severity['severity_level']}"
        )

        print(
            "Base priority      : "
            f"{threat_profile['base_priority']}"
        )

        print(
            "Recommendations     : "
            f"{len(severity['recommendations'])}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "system": "CyberAgent",
        "phase": "Threat Profiling and Severity Engine",
        "description": (
            "Threat profiling and transparent "
            "severity assessment generated from "
            "existing ML investigation evidence."
        ),
        "samples_analyzed": len(results),
        "results": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print("Threat profiling completed.")
    print(
        f"Saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()