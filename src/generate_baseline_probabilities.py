from pathlib import Path
import joblib

import numpy as np


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/predictions")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_model(path):
    return joblib.load(path)


def main():

    print("=" * 70)
    print("GENERATING BASELINE MODEL PROBABILITIES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load final test data
    # --------------------------------------------------------

    network_test = np.load(
        PROCESSED_DIR / "final_network_test.npy"
    )

    dns_test = np.load(
        PROCESSED_DIR / "final_dns_test.npy"
    )

    y_test = np.load(
        PROCESSED_DIR / "final_multiclass_test.npy"
    )

    print("Network test:", network_test.shape)
    print("DNS test:", dns_test.shape)
    print("Labels:", y_test.shape)

    # --------------------------------------------------------
    # Combine network + DNS features
    # --------------------------------------------------------

    X_test = np.concatenate(
        [network_test, dns_test],
        axis=1,
    )

    print("Combined test:", X_test.shape)

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\nLoading Random Forest...")

    rf = load_model(
        "models/baselines/random_forest_multiclass.pkl"
    )

    rf_probabilities = rf.predict_proba(
        X_test
    )

    rf_predictions = np.argmax(
        rf_probabilities,
        axis=1,
    )

    print(
        "RF probabilities:",
        rf_probabilities.shape,
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\nLoading XGBoost...")

    xgb = load_model(
        "models/baselines/xgboost_multiclass.pkl"
    )

    xgb_probabilities = xgb.predict_proba(
        X_test
    )

    xgb_predictions = np.argmax(
        xgb_probabilities,
        axis=1,
    )

    print(
        "XGBoost probabilities:",
        xgb_probabilities.shape,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "random_forest_multiclass_probabilities.npy",
        rf_probabilities,
    )

    np.save(
        OUTPUT_DIR / "random_forest_multiclass_predictions.npy",
        rf_predictions,
    )

    np.save(
        OUTPUT_DIR / "xgboost_multiclass_probabilities.npy",
        xgb_probabilities,
    )

    np.save(
        OUTPUT_DIR / "xgboost_multiclass_predictions.npy",
        xgb_predictions,
    )

    np.save(
        OUTPUT_DIR / "ensemble_multiclass_actual.npy",
        y_test,
    )

    print("\nFiles saved successfully.")


if __name__ == "__main__":
    main()