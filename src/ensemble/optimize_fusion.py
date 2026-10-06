"""
Find the best probability-fusion weights using ONLY
the validation set.

Models:
    Random Forest
    XGBoost
    Original Two-Stream

The test set is NOT used for optimization.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# LOAD VALIDATION PROBABILITIES
# ============================================================

print("=" * 70)
print("VALIDATION-BASED ENSEMBLE WEIGHT OPTIMIZATION")
print("=" * 70)


rf_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "validation_rf_multiclass_probabilities.npy"
    )
)

xgb_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "validation_xgboost_multiclass_probabilities.npy"
    )
)

two_stream_prob = np.load(
    os.path.join(
        PREDICTION_DIR,
        "validation_two_stream_multiclass_probabilities.npy"
    )
)

y_val = np.load(
    os.path.join(
        PREDICTION_DIR,
        "validation_multiclass_actual.npy"
    )
)


print("RF        :", rf_prob.shape)
print("XGBoost   :", xgb_prob.shape)
print("Two-Stream:", two_stream_prob.shape)
print("Labels    :", y_val.shape)


# ============================================================
# VERIFY PROBABILITY SHAPES
# ============================================================

assert rf_prob.shape == xgb_prob.shape
assert rf_prob.shape == two_stream_prob.shape

assert rf_prob.shape[1] == 10


# ============================================================
# HELPER FUNCTION
# ============================================================

def evaluate_weights(
    rf_weight,
    xgb_weight,
    two_stream_weight
):

    ensemble_probability = (
        rf_weight * rf_prob
        + xgb_weight * xgb_prob
        + two_stream_weight * two_stream_prob
    )

    predictions = np.argmax(
        ensemble_probability,
        axis=1
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    weighted_precision = precision_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "rf_weight": rf_weight,
        "xgb_weight": xgb_weight,
        "two_stream_weight": two_stream_weight,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    }


# ============================================================
# GRID SEARCH
# ============================================================

results = []


# We use increments of 0.05.
# The three weights must sum to 1.

weights = np.arange(
    0.0,
    1.01,
    0.05
)


print("\nSearching ensemble weights...")


for rf_weight in weights:

    for xgb_weight in weights:

        two_stream_weight = (
            1.0
            - rf_weight
            - xgb_weight
        )

        # Ignore invalid combinations
        if two_stream_weight < 0:
            continue

        # Avoid floating-point issues
        if two_stream_weight > 1:
            continue

        metrics = evaluate_weights(
            rf_weight,
            xgb_weight,
            two_stream_weight
        )

        results.append(metrics)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# Sort primarily by weighted F1
results_df = results_df.sort_values(
    by="weighted_f1",
    ascending=False
).reset_index(drop=True)


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print("\nTop 15 weight combinations:")
print()

print(
    results_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# BEST WEIGHTS
# ============================================================

best = results_df.iloc[0]


print("\n" + "=" * 70)
print("BEST VALIDATION ENSEMBLE")
print("=" * 70)

print(
    f"Random Forest : "
    f"{best['rf_weight']:.2f}"
)

print(
    f"XGBoost       : "
    f"{best['xgb_weight']:.2f}"
)

print(
    f"Two-Stream    : "
    f"{best['two_stream_weight']:.2f}"
)

print(
    f"\nValidation Accuracy : "
    f"{best['accuracy']:.4f}"
)

print(
    f"Validation Weighted F1 : "
    f"{best['weighted_f1']:.4f}"
)

print(
    f"Validation Macro F1 : "
    f"{best['macro_f1']:.4f}"
)


# ============================================================
# SAVE ALL SEARCH RESULTS
# ============================================================

results_path = os.path.join(
    ENSEMBLE_DIR,
    "fusion_weight_search.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE BEST WEIGHTS
# ============================================================

best_weights = {
    "random_forest": float(
        best["rf_weight"]
    ),
    "xgboost": float(
        best["xgb_weight"]
    ),
    "two_stream": float(
        best["two_stream_weight"]
    )
}


weights_df = pd.DataFrame(
    [best_weights]
)

weights_df.to_csv(
    os.path.join(
        ENSEMBLE_DIR,
        "best_fusion_weights.csv"
    ),
    index=False
)


print(
    "\nSaved:"
)

print(
    "results/ensemble/"
    "fusion_weight_search.csv"
)

print(
    "results/ensemble/"
    "best_fusion_weights.csv"
)