from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

METRIC_DIR = Path("results/metrics")
PRED_DIR = Path("results/predictions")

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
    "Worms",
]

OUTPUT_DIR = Path("results/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_prediction_file(possible_names):
    for name in possible_names:
        path = PRED_DIR / name

        if path.exists():
            return path

    return None


# ============================================================
# LOAD ORIGINAL MODEL RESULTS
# ============================================================

def load_original_results():

    print("=" * 70)
    print("LOADING ORIGINAL TWO-STREAM RESULTS")
    print("=" * 70)

    metrics_path = METRIC_DIR / "two_stream_metrics.json"

    if metrics_path.exists():

        metrics = load_json(metrics_path)

        print("Loaded:", metrics_path)

    else:

        print("Original metrics JSON not found.")
        metrics = None

    binary_actual_path = find_prediction_file([
        "two_stream_binary_actual.npy",
        "binary_actual.npy",
    ])

    binary_prob_path = find_prediction_file([
        "two_stream_binary_probabilities.npy",
        "two_stream_binary_probability.npy",
        "binary_probabilities.npy",
    ])

    binary_pred_path = find_prediction_file([
        "two_stream_binary_predictions.npy",
        "binary_predictions.npy",
    ])

    multi_actual_path = find_prediction_file([
        "two_stream_multiclass_actual.npy",
        "multiclass_actual.npy",
    ])

    multi_pred_path = find_prediction_file([
        "two_stream_multiclass_predictions.npy",
        "multiclass_predictions.npy",
    ])

    if binary_actual_path is None:
        raise FileNotFoundError(
            "Could not find original binary actual predictions."
        )

    if multi_actual_path is None:
        raise FileNotFoundError(
            "Could not find original multiclass actual predictions."
        )

    binary_actual = np.load(binary_actual_path)

    if binary_prob_path is not None:
        binary_probabilities = np.load(binary_prob_path)
    else:
        binary_probabilities = None

    if binary_pred_path is not None:
        binary_predictions = np.load(binary_pred_path)
    else:
        binary_predictions = None

    multiclass_actual = np.load(multi_actual_path)
    multiclass_predictions = np.load(multi_pred_path)

    return {
        "metrics": metrics,
        "binary_actual": binary_actual,
        "binary_probabilities": binary_probabilities,
        "binary_predictions": binary_predictions,
        "multiclass_actual": multiclass_actual,
        "multiclass_predictions": multiclass_predictions,
    }


# ============================================================
# LOAD WEIGHTED MODEL RESULTS
# ============================================================

def load_weighted_results():

    print("\n" + "=" * 70)
    print("LOADING WEIGHTED TWO-STREAM RESULTS")
    print("=" * 70)

    metrics_path = METRIC_DIR / "weighted_two_stream_metrics.json"

    metrics = load_json(metrics_path)

    binary_actual = np.load(
        PRED_DIR / "weighted_binary_actual.npy"
    )

    binary_probabilities = np.load(
        PRED_DIR / "weighted_binary_probabilities.npy"
    )

    binary_predictions = np.load(
        PRED_DIR / "weighted_binary_predictions.npy"
    )

    multiclass_actual = np.load(
        PRED_DIR / "weighted_multiclass_actual.npy"
    )

    multiclass_predictions = np.load(
        PRED_DIR / "weighted_multiclass_predictions.npy"
    )

    return {
        "metrics": metrics,
        "binary_actual": binary_actual,
        "binary_probabilities": binary_probabilities,
        "binary_predictions": binary_predictions,
        "multiclass_actual": multiclass_actual,
        "multiclass_predictions": multiclass_predictions,
    }


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_binary_metrics(results):

    y_true = results["binary_actual"]
    y_pred = results["binary_predictions"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def calculate_multiclass_metrics(results):

    y_true = results["multiclass_actual"]
    y_pred = results["multiclass_predictions"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(original, weighted):

    original_binary = calculate_binary_metrics(original)
    weighted_binary = calculate_binary_metrics(weighted)

    original_multi = calculate_multiclass_metrics(original)
    weighted_multi = calculate_multiclass_metrics(weighted)

    print("\n" + "=" * 70)
    print("BINARY MODEL COMPARISON")
    print("=" * 70)

    binary_df = pd.DataFrame(
        {
            "Original": original_binary,
            "Class Weighted": weighted_binary,
        }
    )

    print(binary_df.round(4))

    print("\n" + "=" * 70)
    print("MULTICLASS MODEL COMPARISON")
    print("=" * 70)

    multi_df = pd.DataFrame(
        {
            "Original": original_multi,
            "Class Weighted": weighted_multi,
        }
    )

    print(multi_df.round(4))

    binary_df.to_csv(
        OUTPUT_DIR / "binary_model_comparison.csv"
    )

    multi_df.to_csv(
        OUTPUT_DIR / "multiclass_model_comparison.csv"
    )

    print("\nComparison files saved.")


# ============================================================
# PER-CLASS COMPARISON
# ============================================================

def create_per_class_comparison(original, weighted):

    original_report = classification_report(
        original["multiclass_actual"],
        original["multiclass_predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    weighted_report = classification_report(
        weighted["multiclass_actual"],
        weighted["multiclass_predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in CLASS_NAMES:

        rows.append(
            {
                "Class": class_name,

                "Original Precision":
                    original_report[class_name]["precision"],

                "Weighted Precision":
                    weighted_report[class_name]["precision"],

                "Precision Change":
                    weighted_report[class_name]["precision"]
                    - original_report[class_name]["precision"],

                "Original Recall":
                    original_report[class_name]["recall"],

                "Weighted Recall":
                    weighted_report[class_name]["recall"],

                "Recall Change":
                    weighted_report[class_name]["recall"]
                    - original_report[class_name]["recall"],

                "Original F1":
                    original_report[class_name]["f1-score"],

                "Weighted F1":
                    weighted_report[class_name]["f1-score"],

                "F1 Change":
                    weighted_report[class_name]["f1-score"]
                    - original_report[class_name]["f1-score"],
            }
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("PER-CLASS COMPARISON")
    print("=" * 70)

    print(df.round(4).to_string(index=False))

    df.to_csv(
        OUTPUT_DIR / "per_class_comparison.csv",
        index=False,
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(original, weighted):

    original_cm = confusion_matrix(
        original["multiclass_actual"],
        original["multiclass_predictions"],
    )

    weighted_cm = confusion_matrix(
        weighted["multiclass_actual"],
        weighted["multiclass_predictions"],
    )

    np.save(
        OUTPUT_DIR / "original_confusion_matrix.npy",
        original_cm,
    )

    np.save(
        OUTPUT_DIR / "weighted_confusion_matrix.npy",
        weighted_cm,
    )

    print("\nConfusion matrices saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    original = load_original_results()

    weighted = load_weighted_results()

    print_comparison(
        original,
        weighted,
    )

    create_per_class_comparison(
        original,
        weighted,
    )

    save_confusion_matrices(
        original,
        weighted,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)

    print(
        "\nResults directory:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()