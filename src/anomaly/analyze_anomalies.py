"""
CyberAgent - Anomaly Score Analysis

This script analyzes the anomaly scores generated
by the Isolation Forest.
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT /
    "results" /
    "anomaly"
)


# ============================================================
# LOAD DATA
# ============================================================

test_scores = np.load(
    RESULT_DIR /
    "isolation_test_anomaly_scores.npy"
)

with open(
    RESULT_DIR /
    "isolation_forest_results.json",
    "r",
    encoding="utf-8"
) as file:

    results = json.load(file)


threshold = results["threshold"]


# ============================================================
# BASIC STATISTICS
# ============================================================

print("=" * 70)
print("CYBERAGENT ANOMALY SCORE ANALYSIS")
print("=" * 70)

print(f"Number of test samples: {len(test_scores):,}")

print(f"\nMinimum score: {test_scores.min():.6f}")
print(f"Maximum score: {test_scores.max():.6f}")
print(f"Mean score:    {test_scores.mean():.6f}")
print(f"Median score:  {np.median(test_scores):.6f}")

print(f"\nThreshold:     {threshold:.6f}")


# ============================================================
# COUNT ANOMALIES
# ============================================================

anomaly_mask = (
    test_scores > threshold
)

normal_count = np.sum(
    ~anomaly_mask
)

anomaly_count = np.sum(
    anomaly_mask
)

print("\n" + "=" * 70)
print("ANOMALY DISTRIBUTION")
print("=" * 70)

print(
    f"Normal-like samples: {normal_count:,}"
)

print(
    f"Anomalous samples:   {anomaly_count:,}"
)

anomaly_percentage = (
    anomaly_count /
    len(test_scores)
) * 100

print(
    f"Anomaly percentage:  "
    f"{anomaly_percentage:.2f}%"
)


# ============================================================
# MOST ANOMALOUS SAMPLES
# ============================================================

top_k = 20

top_indices = np.argsort(
    test_scores
)[-top_k:][::-1]

print("\n" + "=" * 70)
print(f"TOP {top_k} MOST ANOMALOUS TEST SAMPLES")
print("=" * 70)

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"Index={index:<6} "
        f"Score={test_scores[index]:.6f}"
    )