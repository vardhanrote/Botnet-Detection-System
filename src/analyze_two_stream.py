"""
Analysis of the trained Two-Stream model.

This script:
1. Loads saved predictions
2. Creates confusion matrices
3. Calculates per-class performance
4. Finds commonly confused attack classes
5. Checks prediction distribution
6. Checks binary false positives / false negatives
7. Saves analysis results
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


# ==========================================================
# 1. Paths
# ==========================================================

PREDICTIONS_DIR = Path("results/predictions")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# 2. Class names
# ==========================================================

CLASS_NAMES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"
]


# ==========================================================
# 3. Load multiclass predictions
# ==========================================================

y_true = np.load(
    PREDICTIONS_DIR /
    "two_stream_multiclass_actual.npy"
)

y_pred = np.load(
    PREDICTIONS_DIR /
    "two_stream_multiclass_predictions.npy"
)


# ==========================================================
# 4. Confusion matrix
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES))
)


print("\n" + "=" * 70)
print("MULTICLASS CONFUSION MATRIX")
print("=" * 70)

print(
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )
)


# ==========================================================
# 5. Find most common mistakes
# ==========================================================

print("\n" + "=" * 70)
print("MOST COMMON CLASSIFICATION ERRORS")
print("=" * 70)

errors = []

for actual in range(len(CLASS_NAMES)):

    for predicted in range(len(CLASS_NAMES)):

        if actual == predicted:
            continue

        count = cm[actual, predicted]

        if count > 0:

            errors.append(
                (
                    count,
                    CLASS_NAMES[actual],
                    CLASS_NAMES[predicted]
                )
            )


errors.sort(
    reverse=True
)


for count, actual, predicted in errors[:15]:

    print(
        f"{actual:18s} -> "
        f"{predicted:18s}: "
        f"{count} samples"
    )


# ==========================================================
# 6. Per-class metrics
# ==========================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        zero_division=0
    )
)


results = []

for i, class_name in enumerate(CLASS_NAMES):

    results.append(
        {
            "class": class_name,
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": support[i]
        }
    )


metrics_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print(
    metrics_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format
        }
    )
)


# ==========================================================
# 7. Save per-class metrics
# ==========================================================

metrics_df.to_csv(
    METRICS_DIR /
    "two_stream_per_class_metrics.csv",
    index=False
)


# ==========================================================
# 8. Prediction distribution
# ==========================================================

true_counts = np.bincount(
    y_true,
    minlength=len(CLASS_NAMES)
)

pred_counts = np.bincount(
    y_pred,
    minlength=len(CLASS_NAMES)
)


distribution_df = pd.DataFrame(
    {
        "class": CLASS_NAMES,
        "actual_count": true_counts,
        "predicted_count": pred_counts
    }
)


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED DISTRIBUTION")
print("=" * 70)

print(
    distribution_df.to_string(
        index=False
    )
)


distribution_df.to_csv(
    METRICS_DIR /
    "two_stream_prediction_distribution.csv",
    index=False
)


# ==========================================================
# 9. Plot confusion matrix
# ==========================================================

plt.figure(
    figsize=(11, 9)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Two-Stream Model Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

# Add values inside cells
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "two_stream_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================================
# 10. Binary analysis
# ==========================================================

binary_true = np.load(
    PREDICTIONS_DIR /
    "two_stream_binary_actual.npy"
)

binary_pred = np.load(
    PREDICTIONS_DIR /
    "two_stream_binary_predictions.npy"
)


binary_cm = confusion_matrix(
    binary_true,
    binary_pred
)


tn, fp, fn, tp = binary_cm.ravel()


print("\n" + "=" * 70)
print("BINARY DETECTION ANALYSIS")
print("=" * 70)

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


print("\nFalse Positive Rate:")

fpr = fp / (fp + tn)

print(
    f"{fpr:.4f}"
)


print("\nFalse Negative Rate:")

fnr = fn / (fn + tp)

print(
    f"{fnr:.4f}"
)


binary_analysis = {
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "false_positive_rate": float(fpr),
    "false_negative_rate": float(fnr)
}


import json

with open(
    METRICS_DIR /
    "two_stream_binary_analysis.json",
    "w"
) as file:

    json.dump(
        binary_analysis,
        file,
        indent=4
    )


# ==========================================================
# 11. Final summary
# ==========================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    "- two_stream_per_class_metrics.csv"
)

print(
    "- two_stream_prediction_distribution.csv"
)

print(
    "- two_stream_binary_analysis.json"
)

print(
    "- two_stream_confusion_matrix.png"
)