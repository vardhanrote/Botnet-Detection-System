from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


PRED_DIR = Path("results/predictions")
OUTPUT_DIR = Path("results/comparison")

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


def analyze_errors(
    model_name,
    actual_file,
    prediction_file,
):

    actual = np.load(
        PRED_DIR / actual_file
    )

    predictions = np.load(
        PRED_DIR / prediction_file
    )

    cm = confusion_matrix(
        actual,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
    )

    errors = []

    for actual_id in range(len(CLASS_NAMES)):

        for predicted_id in range(len(CLASS_NAMES)):

            if actual_id == predicted_id:
                continue

            count = cm[
                actual_id,
                predicted_id,
            ]

            if count > 0:

                errors.append(
                    {
                        "Actual": CLASS_NAMES[actual_id],
                        "Predicted": CLASS_NAMES[predicted_id],
                        "Errors": int(count),
                    }
                )

    error_df = pd.DataFrame(errors)

    error_df = error_df.sort_values(
        "Errors",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        error_df.head(20).to_string(
            index=False
        )
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_top_errors.csv"
    )

    error_df.to_csv(
        OUTPUT_DIR / filename,
        index=False,
    )


def main():

    analyze_errors(
        "Original Two Stream",
        "two_stream_multiclass_actual.npy",
        "two_stream_multiclass_predictions.npy",
    )

    analyze_errors(
        "Weighted Two Stream",
        "weighted_multiclass_actual.npy",
        "weighted_multiclass_predictions.npy",
    )


if __name__ == "__main__":
    main()