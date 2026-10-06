"""
CyberAgent - LIME Local Explainability

This module explains individual Random Forest predictions.

SHAP:
    Global + local feature contribution analysis.

LIME:
    Local explanation for one individual prediction.

The goal is to provide an analyst-friendly explanation
for a suspicious network event.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

from src.explainability.feature_names import (
    ALL_FEATURES,
    ATTACK_CLASSES,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "baselines" /
    "random_forest_multiclass.pkl"
)

RESULT_DIR = (
    PROJECT_ROOT /
    "results" /
    "explainability"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Number of LIME perturbations.
#
# 3000 is enough for a useful demonstration while
# keeping computation reasonable.
NUM_SAMPLES = 3000


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data():

    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    network_test = np.load(
        DATA_DIR /
        "final_network_test.npy"
    )

    dns_test = np.load(
        DATA_DIR /
        "final_dns_test.npy"
    )

    y_test = np.load(
        DATA_DIR /
        "final_multiclass_test.npy"
    )

    # Combine the two streams.
    X_test = np.concatenate(
        [
            network_test,
            dns_test
        ],
        axis=1
    )

    print(
        f"Test data shape: {X_test.shape}"
    )

    return X_test, y_test


# ============================================================
# 4. LOAD MODEL
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
# 5. CREATE LIME EXPLAINER
# ============================================================

def create_explainer(
    X_train_reference
):

    """
    LIME needs a representative dataset to understand
    the distribution of each feature.

    We use the final training data as the reference.
    """

    print("\n" + "=" * 70)
    print("CREATING LIME EXPLAINER")
    print("=" * 70)

    explainer = LimeTabularExplainer(
        training_data=X_train_reference,
        feature_names=ALL_FEATURES,
        class_names=ATTACK_CLASSES,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    print("LIME explainer created.")

    return explainer


# ============================================================
# 6. FIND A SUSPICIOUS SAMPLE
# ============================================================

def find_attack_sample(
    model,
    X_test,
    y_test
):

    """
    Select a correctly classified attack sample.

    This gives us a meaningful example for the demo.
    """

    predictions = model.predict(
        X_test
    )

    # Find attack samples that the model
    # correctly classified.
    candidates = np.where(
        (y_test != 0) &
        (predictions == y_test)
    )[0]

    if len(candidates) == 0:

        print(
            "No correctly classified attack sample found."
        )

        # Fall back to the first attack.
        candidates = np.where(
            y_test != 0
        )[0]

    index = int(
        candidates[0]
    )

    print("\n" + "=" * 70)
    print("SELECTED SAMPLE")
    print("=" * 70)

    print(f"Sample index: {index}")
    print(
        f"Actual class: "
        f"{ATTACK_CLASSES[y_test[index]]}"
    )

    print(
        f"Predicted class: "
        f"{ATTACK_CLASSES[predictions[index]]}"
    )

    return index


# ============================================================
# 7. CREATE LOCAL EXPLANATION
# ============================================================

def explain_sample(
    explainer,
    model,
    X_test,
    sample_index
):

    """
    Generate a local LIME explanation.
    """

    print("\n" + "=" * 70)
    print("GENERATING LIME EXPLANATION")
    print("=" * 70)

    sample = X_test[
        sample_index
    ]

    probabilities = model.predict_proba(
        sample.reshape(1, -1)
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=10,
        top_labels=3,
        num_samples=NUM_SAMPLES,
    )

    print(
        "LIME explanation generated."
    )

    print(
        f"\nPredicted class: "
        f"{ATTACK_CLASSES[predicted_class]}"
    )

    print(
        f"Prediction confidence: "
        f"{probabilities[predicted_class]:.4f}"
    )

    return explanation, probabilities, predicted_class


# ============================================================
# 8. DISPLAY EXPLANATION
# ============================================================

def display_explanation(
    explanation,
    predicted_class
):

    print("\n" + "=" * 70)
    print("LIME FEATURE CONTRIBUTIONS")
    print("=" * 70)

    contributions = (
        explanation.as_list(
            label=predicted_class
        )
    )

    for rank, (
        feature,
        weight
    ) in enumerate(
        contributions,
        start=1
    ):

        direction = (
            "supports prediction"
            if weight > 0
            else "opposes prediction"
        )

        print(
            f"{rank:2d}. "
            f"{feature:<45} "
            f"{weight:+.6f} "
            f"({direction})"
        )

    return contributions


# ============================================================
# 9. SAVE JSON EXPLANATION
# ============================================================

def save_explanation(
    sample_index,
    actual_class,
    predicted_class,
    confidence,
    contributions,
    sample_values
):

    output = {

        "sample_index": int(
            sample_index
        ),

        "actual_class": actual_class,

        "predicted_class": predicted_class,

        "confidence": float(
            confidence
        ),

        "features": {
            feature: float(value)
            for feature, value
            in zip(
                ALL_FEATURES,
                sample_values
            )
        },

        "lime_explanation": [
            {
                "feature": feature,
                "weight": float(weight),
                "effect": (
                    "supports_prediction"
                    if weight > 0
                    else "opposes_prediction"
                ),
            }
            for feature, weight
            in contributions
        ],
    }

    output_path = (
        RESULT_DIR /
        "lime_sample_explanation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        f"\nExplanation saved to:"
        f"\n{output_path}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CYBERAGENT - LIME LOCAL EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load training data for LIME reference
    # --------------------------------------------------------

    network_train = np.load(
        DATA_DIR /
        "final_network_train.npy"
    )

    dns_train = np.load(
        DATA_DIR /
        "final_dns_train.npy"
    )

    X_train_reference = np.concatenate(
        [
            network_train,
            dns_train
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Create explainer
    # --------------------------------------------------------

    explainer = create_explainer(
        X_train_reference
    )

    # --------------------------------------------------------
    # Find useful sample
    # --------------------------------------------------------

    sample_index = find_attack_sample(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    (
        explanation,
        probabilities,
        predicted_class
    ) = explain_sample(
        explainer,
        model,
        X_test,
        sample_index
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    contributions = display_explanation(
        explanation,
        predicted_class
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_explanation(
        sample_index=sample_index,
        actual_class=ATTACK_CLASSES[
            y_test[sample_index]
        ],
        predicted_class=ATTACK_CLASSES[
            predicted_class
        ],
        confidence=probabilities[
            predicted_class
        ],
        contributions=contributions,
        sample_values=X_test[
            sample_index
        ],
    )

    print("\n" + "=" * 70)
    print("LIME ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()