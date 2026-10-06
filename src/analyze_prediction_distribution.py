from pathlib import Path
import numpy as np
import pandas as pd


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


def analyze_model(
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

    actual_counts = np.bincount(
        actual,
        minlength=len(CLASS_NAMES),
    )

    prediction_counts = np.bincount(
        predictions,
        minlength=len(CLASS_NAMES),
    )

    df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Actual Count": actual_counts,
            "Predicted Count": prediction_counts,
            "Difference":
                prediction_counts - actual_counts,
        }
    )

    df["Prediction / Actual"] = (
        df["Predicted Count"]
        / df["Actual Count"].replace(0, np.nan)
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        df.round(2).to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR
        / f"{model_name.lower().replace(' ', '_')}_distribution.csv",
        index=False,
    )


def main():

    analyze_model(
        "Original Two Stream",
        "two_stream_multiclass_actual.npy",
        "two_stream_multiclass_predictions.npy",
    )

    analyze_model(
        "Weighted Two Stream",
        "weighted_multiclass_actual.npy",
        "weighted_multiclass_predictions.npy",
    )


if __name__ == "__main__":
    main()