"""
Run CyberAgent Evidence Extraction Engine
on real samples from the UNSW-NB15 test set.
"""

import json
from pathlib import Path

import numpy as np

from src.investigation.evidence_extractor import (
    EvidenceExtractor
)


def main():

    # ---------------------------------------------------------
    # Project paths
    # ---------------------------------------------------------

    project_root = Path(
        __file__
    ).resolve().parents[2]

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    output_dir = (
        project_root
        / "results"
        / "investigation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load final test data
    # ---------------------------------------------------------

    network_test = np.load(
        processed_dir
        / "final_network_test.npy"
    )

    dns_test = np.load(
        processed_dir
        / "final_dns_test.npy"
    )

    print("=" * 70)
    print("CYBERAGENT EVIDENCE EXTRACTION ENGINE")
    print("=" * 70)

    print(
        f"Network test shape: {network_test.shape}"
    )

    print(
        f"DNS test shape:     {dns_test.shape}"
    )

    # ---------------------------------------------------------
    # Create engine
    # ---------------------------------------------------------

    print("\nLoading models...")

    engine = EvidenceExtractor(
        project_root=project_root
    )

    print("Models loaded successfully.")

    # ---------------------------------------------------------
    # Select samples
    # ---------------------------------------------------------

    sample_indices = [
        0,
        100,
        1000,
        5000,
        10000,
    ]

    all_results = []

    # ---------------------------------------------------------
    # Process samples
    # ---------------------------------------------------------

    for sample_index in sample_indices:

        print("\n" + "-" * 70)
        print(
            f"Analyzing sample {sample_index}"
        )
        print("-" * 70)

        evidence = engine.extract(
            network_features=network_test[
                sample_index
            ],
            dns_features=dns_test[
                sample_index
            ],
            sample_index=sample_index,
        )

        all_results.append(evidence)

        # -----------------------------------------------------
        # Display important information
        # -----------------------------------------------------

        binary = evidence[
            "binary_detection"
        ]

        classification = evidence[
            "attack_classification"
        ]

        anomaly = evidence[
            "anomaly_detection"
        ]

        consistency = evidence[
            "model_consistency"
        ]

        strength = evidence[
            "evidence_strength"
        ]

        print(
            f"Binary result       : "
            f"{binary['prediction']}"
        )

        print(
            f"Attack probability  : "
            f"{binary['attack_probability']:.4f}"
        )

        print(
            f"Random Forest class : "
            f"{classification['random_forest']['predicted_class']}"
        )

        print(
            f"RF confidence       : "
            f"{classification['random_forest']['confidence']:.4f}"
        )

        print(
            f"Two-Stream class    : "
            f"{classification['two_stream']['multiclass_prediction']}"
        )

        print(
            f"Anomaly detected    : "
            f"{anomaly['is_anomaly']}"
        )

        print(
            f"Anomaly score       : "
            f"{anomaly['anomaly_score']:.6f}"
        )

        print(
            f"Class agreement     : "
            f"{consistency['agreement_status']}"
        )

        print(
            f"Evidence strength   : "
            f"{strength['level']} "
            f"({strength['score']}/100)"
        )

        print(
            f"Investigation       : "
            f"Evidence extraction completed"
        )

    # ---------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------

    output_path = (
        output_dir
        / "evidence_extraction_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("Evidence extraction completed.")
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()