from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


PRED_DIR = Path("results/predictions")
OUTPUT_DIR = Path("results/ensemble")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


# Initial interpretable weights
RF_WEIGHT = 0.40
XGB_WEIGHT = 0.30
TWO_STREAM_WEIGHT = 0.30


def main():

    print("=" * 70)
    print("CYBERAGENT MULTICLASS ENSEMBLE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load probabilities
    # --------------------------------------------------------

    rf_prob = np.load(
        PRED_DIR
        / "random_forest_multiclass_probabilities.npy"
    )

    xgb_prob = np.load(
        PRED_DIR
        / "xgboost_multiclass_probabilities.npy"
    )

    two_stream_prob = np.load(
        PRED_DIR
        / "two_stream_multiclass_probabilities.npy"
    )

    y_true = np.load(
        PRED_DIR
        / "ensemble_multiclass_actual.npy"
    )

    print("RF:", rf_prob.shape)
    print("XGBoost:", xgb_prob.shape)
    print("Two-Stream:", two_stream_prob.shape)

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if not (
        rf_prob.shape
        == xgb_prob.shape
        == two_stream_prob.shape
    ):
        raise ValueError(
            "Model probability shapes do not match."
        )

    if rf_prob.shape[1] != 10:
        raise ValueError(
            "Expected 10 attack classes."
        )

    # --------------------------------------------------------
    # Probability fusion
    # --------------------------------------------------------

    ensemble_prob = (
        RF_WEIGHT * rf_prob
        + XGB_WEIGHT * xgb_prob
        + TWO_STREAM_WEIGHT * two_stream_prob
    )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    ensemble_prediction = np.argmax(
        ensemble_prob,
        axis=1,
    )

    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    rf_prediction = np.argmax(
        rf_prob,
        axis=1,
    )

    xgb_prediction = np.argmax(
        xgb_prob,
        axis=1,
    )

    two_stream_prediction = np.argmax(
        two_stream_prob,
        axis=1,
    )

    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    agreement_count = (
        (rf_prediction == ensemble_prediction).astype(int)
        +
        (xgb_prediction == ensemble_prediction).astype(int)
        +
        (two_stream_prediction == ensemble_prediction).astype(int)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = np.max(
        ensemble_prob,
        axis=1,
    )

    # --------------------------------------------------------
    # Entropy-based uncertainty
    # --------------------------------------------------------

    safe_prob = np.clip(
        ensemble_prob,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        safe_prob * np.log(safe_prob),
        axis=1,
    )

    max_entropy = np.log(
        ensemble_prob.shape[1]
    )

    normalized_uncertainty = (
        entropy / max_entropy
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "ensemble_multiclass_probabilities.npy",
        ensemble_prob,
    )

    np.save(
        OUTPUT_DIR / "ensemble_multiclass_predictions.npy",
        ensemble_prediction,
    )

    np.save(
        OUTPUT_DIR / "ensemble_confidence.npy",
        confidence,
    )

    np.save(
        OUTPUT_DIR / "ensemble_uncertainty.npy",
        normalized_uncertainty,
    )

    np.save(
        OUTPUT_DIR / "ensemble_agreement.npy",
        agreement_count,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        ensemble_prediction,
    )

    precision = precision_score(
        y_true,
        ensemble_prediction,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        ensemble_prediction,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        ensemble_prediction,
        average="weighted",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        ensemble_prediction,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Weighted F1    : {f1_weighted:.4f}"
    )

    print(
        f"Macro F1       : {f1_macro:.4f}"
    )

    print(
        f"Weighted Prec. : {precision:.4f}"
    )

    print(
        f"Weighted Recall: {recall:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        ensemble_prediction,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Detailed sample-level report
    # --------------------------------------------------------

    detailed_df = pd.DataFrame(
        {
            "Actual": [
                CLASS_NAMES[i]
                for i in y_true
            ],

            "RF Prediction": [
                CLASS_NAMES[i]
                for i in rf_prediction
            ],

            "XGBoost Prediction": [
                CLASS_NAMES[i]
                for i in xgb_prediction
            ],

            "Two-Stream Prediction": [
                CLASS_NAMES[i]
                for i in two_stream_prediction
            ],

            "Ensemble Prediction": [
                CLASS_NAMES[i]
                for i in ensemble_prediction
            ],

            "Confidence": confidence,

            "Uncertainty": normalized_uncertainty,

            "Agreement": agreement_count,
        }
    )

    detailed_df.to_csv(
        OUTPUT_DIR
        / "ensemble_sample_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        ensemble_prediction,
    )

    np.save(
        OUTPUT_DIR
        / "ensemble_confusion_matrix.npy",
        cm,
    )

    print(
        "\nAll ensemble results saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()