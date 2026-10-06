"""
Evaluate the validation-selected ensemble weights on the
UNSEEN test set.

IMPORTANT:
Weights are already frozen.
No test-set optimization happens here.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_DIR = "results/predictions"
ENSEMBLE_DIR = "results/ensemble"

os.makedirs(
    ENSEMBLE_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST PROBABILITIES
# ============================================================

print("=" * 70)
print("FINAL FROZEN ENSEMBLE EVALUATION")
print("=" * 70)


rf_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "random_forest_multiclass_probabilities.npy"
    )
)

xgb_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "xgboost_multiclass_probabilities.npy"
    )
)

two_stream_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "two_stream_multiclass_probabilities.npy"
    )
)

y_test = np.load(
    os.path.join(
        PREDICTION_DIR,
        "ensemble_multiclass_actual.npy"
    )
)


# ============================================================
# LOAD FROZEN WEIGHTS
# ============================================================

weights_df = pd.read_csv(
    os.path.join(
        ENSEMBLE_DIR,
        "best_fusion_weights.csv"
    )
)

rf_weight = weights_df.loc[
    0,
    "random_forest"
]

xgb_weight = weights_df.loc[
    0,
    "xgboost"
]

two_stream_weight = weights_df.loc[
    0,
    "two_stream"
]


print("\nFrozen weights:")

print(
    f"Random Forest : {rf_weight:.2f}"
)

print(
    f"XGBoost       : {xgb_weight:.2f}"
)

print(
    f"Two-Stream    : {two_stream_weight:.2f}"
)


# ============================================================
# FUSE PROBABILITIES
# ============================================================

ensemble_probability = (
    rf_weight * rf_prob
    + xgb_weight * xgb_prob
    + two_stream_weight * two_stream_prob
)


# ============================================================
# PREDICTIONS
# ============================================================

ensemble_prediction = np.argmax(
    ensemble_probability,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)

weighted_precision = precision_score(
    y_test,
    ensemble_prediction,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    ensemble_prediction,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    ensemble_prediction,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    ensemble_prediction,
    average="macro",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Weighted F1    : {weighted_f1:.4f}"
)

print(
    f"Macro F1       : {macro_f1:.4f}"
)

print(
    f"Weighted Prec. : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall: {weighted_recall:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
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


print("\nClassification Report:")

report = classification_report(
    y_test,
    ensemble_prediction,
    target_names=class_names,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ensemble_prediction
)

np.save(
    os.path.join(
        ENSEMBLE_DIR,
        "final_ensemble_confusion_matrix.npy"
    ),
    cm
)


# ============================================================
# SAVE PROBABILITIES
# ============================================================

np.save(
    os.path.join(
        ENSEMBLE_DIR,
        "final_ensemble_probabilities.npy"
    ),
    ensemble_probability
)

np.save(
    os.path.join(
        ENSEMBLE_DIR,
        "final_ensemble_predictions.npy"
    ),
    ensemble_prediction
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "rf_weight": rf_weight,
        "xgb_weight": xgb_weight,
        "two_stream_weight": two_stream_weight
    }
])

metrics.to_csv(
    os.path.join(
        ENSEMBLE_DIR,
        "final_ensemble_metrics.csv"
    ),
    index=False
)


print("\nFinal ensemble files saved.")