"""
CyberAgent - SHAP Explainability

This script explains the Random Forest multiclass model.

The Random Forest is our selected multiclass model because
it achieved the strongest held-out multiclass performance
among the tested models.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.explainability.feature_names import (
    ALL_FEATURES,
    ATTACK_CLASSES,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baselines"
    / "random_forest_multiclass.pkl"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "explainability"
)

FIGURE_DIR = (
    RESULT_DIR
    / "figures"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# We don't need to explain all 82k test samples.
#
# A representative sample is enough for global explanation
# and keeps SHAP computation manageable.

SAMPLE_SIZE = 1000

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING DATA FOR SHAP")
    print("=" * 70)

    network_test = np.load(
        DATA_DIR / "final_network_test.npy"
    )

    dns_test = np.load(
        DATA_DIR / "final_dns_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "final_multiclass_test.npy"
    )

    # Combine network and DNS streams
    X_test = np.concatenate(
        [network_test, dns_test],
        axis=1
    )

    print(f"Test features: {X_test.shape}")
    print(f"Test labels:   {y_test.shape}")

    return X_test, y_test


# ============================================================
# 4. LOAD RANDOM FOREST
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING RANDOM FOREST")
    print("=" * 70)

    model = joblib.load(
        MODEL_PATH
    )

    print("Random Forest loaded.")

    return model


# ============================================================
# 5. SELECT REPRESENTATIVE SAMPLES
# ============================================================

def select_samples(X_test, y_test):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(X_test)
    )

    indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False
    )

    X_sample = X_test[indices]
    y_sample = y_test[indices]

    print("\n" + "=" * 70)
    print("SELECTED SHAP SAMPLE")
    print("=" * 70)

    print(
        f"Selected {len(X_sample):,} samples."
    )

    return X_sample, y_sample


# ============================================================
# 6. CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(
    model,
    X_sample
):

    print("\n" + "=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    # TreeExplainer is optimized for tree-based models
    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    print("SHAP values calculated.")

    return explainer, shap_values


# ============================================================
# 7. HANDLE SHAP OUTPUT FORMAT
# ============================================================

def convert_shap_values(shap_values):

    """
    SHAP can return different structures depending
    on the installed version.

    We convert the result into:

        samples × features × classes
    """

    if isinstance(
        shap_values,
        list
    ):

        # Older SHAP format:
        # list[class] -> samples × features

        array = np.stack(
            shap_values,
            axis=2
        )

        return array

    array = np.asarray(
        shap_values
    )

    # Some versions return:
    #
    # samples × features × classes
    #
    # while others can return:
    #
    # samples × classes × features

    if array.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP shape: {array.shape}"
        )

    # If second dimension matches number of classes
    # and third dimension matches number of features,
    # transpose it.

    if (
        array.shape[1] == len(ATTACK_CLASSES)
        and array.shape[2] == len(ALL_FEATURES)
    ):

        array = np.transpose(
            array,
            (0, 2, 1)
        )

    return array


# ============================================================
# 8. GLOBAL FEATURE IMPORTANCE
# ============================================================

def calculate_global_importance(
    shap_values
):

    """
    Calculate mean absolute SHAP value
    across samples and classes.
    """

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=(0, 2)
    )

    importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "mean_absolute_shap": mean_abs_shap,
        }
    )

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    return importance


# ============================================================
# 9. SAVE GLOBAL IMPORTANCE
# ============================================================

def save_global_importance(
    importance
):

    output_path = (
        RESULT_DIR
        / "shap_global_feature_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nGlobal SHAP importance saved to:"
        f"\n{output_path}"
    )

    print("\nTop features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )


# ============================================================
# 10. CREATE SHAP BAR PLOT
# ============================================================

def create_bar_plot(
    importance
):

    plt.figure(
        figsize=(10, 6)
    )

    plot_data = importance.sort_values(
        "mean_absolute_shap"
    )

    plt.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap"]
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "CyberAgent - Global SHAP Feature Importance"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "shap_global_feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SHAP bar plot saved to:"
        f"\n{output_path}"
    )


# ============================================================
# 11. SAVE RAW SHAP VALUES
# ============================================================

def save_shap_values(
    shap_values
):

    output_path = (
        RESULT_DIR
        / "shap_values.npy"
    )

    np.save(
        output_path,
        shap_values
    )

    print(
        f"SHAP values saved to:"
        f"\n{output_path}"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CYBERAGENT - RANDOM FOREST SHAP EXPLAINABILITY")
    print("=" * 70)

    # Load test data
    X_test, y_test = load_data()

    # Load model
    model = load_model()

    # Select samples
    X_sample, y_sample = select_samples(
        X_test,
        y_test
    )

    # SHAP
    explainer, shap_values = (
        calculate_shap_values(
            model,
            X_sample
        )
    )

    # Convert format
    shap_values = convert_shap_values(
        shap_values
    )

    print(
        f"\nFinal SHAP shape: "
        f"{shap_values.shape}"
    )

    # Global importance
    importance = calculate_global_importance(
        shap_values
    )

    # Save results
    save_global_importance(
        importance
    )

    create_bar_plot(
        importance
    )

    save_shap_values(
        shap_values
    )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()