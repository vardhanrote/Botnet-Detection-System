import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# CREATE RESULT DIRECTORIES
# --------------------------------------------------

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    average="weighted"
):
    """
    Calculate standard classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# --------------------------------------------------
# PRINT METRICS
# --------------------------------------------------

def print_metrics(
    model_name,
    metrics
):

    print("\n" + "=" * 60)

    print(
        f"{model_name} RESULTS"
    )

    print("=" * 60)

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1-score  : {metrics['f1_score']:.4f}"
    )


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

def save_metrics(
    model_name,
    task,
    metrics,
    training_time=None
):

    result = {
        "model": model_name,
        "task": task,
        **metrics
    }

    if training_time is not None:

        result[
            "training_time_seconds"
        ] = training_time

    file_name = (
        f"results/metrics/"
        f"{model_name}_{task}.json"
    )

    with open(
        file_name,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to: {file_name}"
    )


# --------------------------------------------------
# SAVE CONFUSION MATRIX
# --------------------------------------------------

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    task,
    class_names=None
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        f"{model_name} - {task} Confusion Matrix"
    )

    plt.tight_layout()

    file_name = (
        f"results/figures/"
        f"{model_name}_{task}_confusion_matrix.png"
    )

    plt.savefig(
        file_name,
        dpi=300
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {file_name}"
    )


# --------------------------------------------------
# PRINT CLASSIFICATION REPORT
# --------------------------------------------------

def print_classification_report(
    y_true,
    y_pred,
    class_names=None
):

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )