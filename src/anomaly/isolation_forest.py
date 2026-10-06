"""
CyberAgent - Isolation Forest Anomaly Detection

This module trains an Isolation Forest using NORMAL traffic only.

Purpose:
    Learn the normal behavior of network + DNS features and
    identify unusual traffic as anomalies.

Important:
    The anomaly detector is NOT trained using attack samples.
"""

from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "anomaly"
RESULT_DIR = PROJECT_ROOT / "results" / "anomaly"


# Create folders if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Number of trees in Isolation Forest
N_ESTIMATORS = 200

# Maximum number of samples used by each tree
MAX_SAMPLES = 10000

# Use all available CPU cores
N_JOBS = -1

# Maximum number of normal samples used for training.
#
# We have many normal samples, so limiting this keeps
# training reasonably fast on a CPU.
MAX_NORMAL_SAMPLES = 100000


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    """
    Load the final processed network, DNS and multiclass data.
    """

    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    network_train = np.load(
        DATA_DIR / "final_network_train.npy"
    )

    network_val = np.load(
        DATA_DIR / "final_network_val.npy"
    )

    network_test = np.load(
        DATA_DIR / "final_network_test.npy"
    )

    dns_train = np.load(
        DATA_DIR / "final_dns_train.npy"
    )

    dns_val = np.load(
        DATA_DIR / "final_dns_val.npy"
    )

    dns_test = np.load(
        DATA_DIR / "final_dns_test.npy"
    )

    y_train = np.load(
        DATA_DIR / "final_multiclass_train.npy"
    )

    y_val = np.load(
        DATA_DIR / "final_multiclass_val.npy"
    )

    y_test = np.load(
        DATA_DIR / "final_multiclass_test.npy"
    )

    print(f"Network train: {network_train.shape}")
    print(f"DNS train:     {dns_train.shape}")

    print(f"Network val:   {network_val.shape}")
    print(f"DNS val:       {dns_val.shape}")

    print(f"Network test:  {network_test.shape}")
    print(f"DNS test:      {dns_test.shape}")

    return (
        network_train,
        network_val,
        network_test,
        dns_train,
        dns_val,
        dns_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# 4. COMBINE NETWORK + DNS FEATURES
# ============================================================

def combine_features(network, dns):
    """
    Combine the two feature streams into one 10-feature vector.

    Network:
        5 features

    DNS:
        5 features

    Final:
        10 features
    """

    return np.concatenate(
        [network, dns],
        axis=1
    )


# ============================================================
# 5. TRAIN ISOLATION FOREST
# ============================================================

def train_anomaly_detector(X_train, y_train):
    """
    Train Isolation Forest ONLY on normal traffic.

    Normal class:
        0 = Normal
    """

    print("\n" + "=" * 70)
    print("PREPARING NORMAL TRAFFIC")
    print("=" * 70)

    # Normal class is encoded as 0
    normal_mask = y_train == 0

    X_normal = X_train[normal_mask]

    print(f"Total training samples: {len(X_train):,}")
    print(f"Normal training samples: {len(X_normal):,}")

    # --------------------------------------------------------
    # Limit training size if necessary
    # --------------------------------------------------------

    if len(X_normal) > MAX_NORMAL_SAMPLES:

        rng = np.random.default_rng(RANDOM_STATE)

        selected_indices = rng.choice(
            len(X_normal),
            size=MAX_NORMAL_SAMPLES,
            replace=False
        )

        X_normal = X_normal[selected_indices]

        print(
            f"Using {MAX_NORMAL_SAMPLES:,} normal samples "
            "for Isolation Forest training."
        )

    else:
        print("Using all available normal samples.")

    print(f"Final anomaly-training shape: {X_normal.shape}")

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(X_normal)

    print("Isolation Forest training completed.")

    return model, X_normal


# ============================================================
# 6. CALCULATE ANOMALY SCORES
# ============================================================

def calculate_anomaly_scores(model, X):
    """
    Calculate anomaly scores.

    Isolation Forest's decision_function:
        positive -> more normal
        negative -> more anomalous

    We reverse the value so:

        HIGHER SCORE = MORE ANOMALOUS
    """

    decision_scores = model.decision_function(X)

    anomaly_scores = -decision_scores

    return anomaly_scores


# ============================================================
# 7. SELECT ANOMALY THRESHOLD
# ============================================================

def calculate_threshold(model, X_normal):
    """
    Calculate a threshold using normal training traffic.

    We use the 95th percentile of anomaly scores.

    Therefore approximately 95% of normal training traffic
    should fall below the threshold.
    """

    normal_scores = calculate_anomaly_scores(
        model,
        X_normal
    )

    threshold = float(
        np.percentile(normal_scores, 95)
    )

    print("\n" + "=" * 70)
    print("ANOMALY THRESHOLD")
    print("=" * 70)

    print(f"Threshold: {threshold:.6f}")

    print(
        "Traffic with anomaly score above this value "
        "will be marked as anomalous."
    )

    return threshold


# ============================================================
# 8. CONVERT SCORE TO NORMAL / ANOMALY
# ============================================================

def predict_anomaly(anomaly_scores, threshold):
    """
    Convert anomaly scores into binary predictions.

    0 = Normal
    1 = Anomaly
    """

    predictions = (
        anomaly_scores > threshold
    ).astype(int)

    return predictions


# ============================================================
# 9. EVALUATE ANOMALY DETECTOR
# ============================================================

def evaluate_detector(
    name,
    y_true_multiclass,
    anomaly_scores,
    threshold
):
    """
    Evaluate anomaly detection.

    For anomaly detection:

        Normal = 0
        Attack = 1
    """

    # Convert multiclass labels into binary labels
    #
    # 0 -> Normal
    # 1-9 -> Attack

    y_true = (
        y_true_multiclass != 0
    ).astype(int)

    y_pred = predict_anomaly(
        anomaly_scores,
        threshold
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print(f"{name.upper()} ANOMALY DETECTION RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly"
            ],
            zero_division=0
        )
    )

    # Calculate FPR and FNR
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    model,
    threshold,
    train_scores,
    val_scores,
    test_scores,
    val_metrics,
    test_metrics,
):
    """
    Save the model, anomaly scores and metrics.
    """

    print("\n" + "=" * 70)
    print("SAVING ANOMALY DETECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "isolation_forest_normal.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to: {model_path}")

    # --------------------------------------------------------
    # Save anomaly scores
    # --------------------------------------------------------

    np.save(
        RESULT_DIR /
        "isolation_train_anomaly_scores.npy",
        train_scores
    )

    np.save(
        RESULT_DIR /
        "isolation_val_anomaly_scores.npy",
        val_scores
    )

    np.save(
        RESULT_DIR /
        "isolation_test_anomaly_scores.npy",
        test_scores
    )

    # --------------------------------------------------------
    # Save configuration + metrics
    # --------------------------------------------------------

    results = {
        "model": "Isolation Forest",
        "training_strategy": (
            "Trained only on normal traffic"
        ),
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "threshold_percentile": 95,
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    results_path = (
        RESULT_DIR /
        "isolation_forest_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(f"Results saved to: {results_path}")


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CYBERAGENT - ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        network_train,
        network_val,
        network_test,
        dns_train,
        dns_val,
        dns_test,
        y_train,
        y_val,
        y_test,
    ) = load_data()

    # --------------------------------------------------------
    # Combine network + DNS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING NETWORK + DNS FEATURES")
    print("=" * 70)

    X_train = combine_features(
        network_train,
        dns_train
    )

    X_val = combine_features(
        network_val,
        dns_val
    )

    X_test = combine_features(
        network_test,
        dns_test
    )

    print(f"Combined train: {X_train.shape}")
    print(f"Combined val:   {X_val.shape}")
    print(f"Combined test:  {X_test.shape}")

    # --------------------------------------------------------
    # Train anomaly detector
    # --------------------------------------------------------

    model, X_normal = train_anomaly_detector(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = calculate_threshold(
        model,
        X_normal
    )

    # --------------------------------------------------------
    # Generate anomaly scores
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING ANOMALY SCORES")
    print("=" * 70)

    train_scores = calculate_anomaly_scores(
        model,
        X_train
    )

    val_scores = calculate_anomaly_scores(
        model,
        X_val
    )

    test_scores = calculate_anomaly_scores(
        model,
        X_test
    )

    print("Train anomaly scores generated.")
    print("Validation anomaly scores generated.")
    print("Test anomaly scores generated.")

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    val_metrics = evaluate_detector(
        "Validation",
        y_val,
        val_scores,
        threshold
    )

    # --------------------------------------------------------
    # Evaluate test
    # --------------------------------------------------------

    test_metrics = evaluate_detector(
        "Test",
        y_test,
        test_scores,
        threshold
    )

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------

    save_results(
        model,
        threshold,
        train_scores,
        val_scores,
        test_scores,
        val_metrics,
        test_metrics,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ISOLATION FOREST COMPLETE")
    print("=" * 70)

    print(f"Test F1:      {test_metrics['f1']:.4f}")
    print(f"Test Recall:  {test_metrics['recall']:.4f}")
    print(f"Test FPR:     {test_metrics['fpr']:.4f}")

    print("\nSaved model:")
    print(
        "models/anomaly/isolation_forest_normal.pkl"
    )

    print("\nSaved results:")
    print(
        "results/anomaly/isolation_forest_results.json"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()