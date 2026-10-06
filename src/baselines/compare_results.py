import os
import json
import pandas as pd


# --------------------------------------------------
# RESULT DIRECTORY
# --------------------------------------------------

metrics_dir = "results/metrics"


# --------------------------------------------------
# READ ALL JSON FILES
# --------------------------------------------------

results = []


for file_name in os.listdir(metrics_dir):

    if file_name.endswith(".json"):

        file_path = os.path.join(
            metrics_dir,
            file_name
        )

        with open(
            file_path,
            "r"
        ) as file:

            result = json.load(
                file
            )

        results.append(
            result
        )


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(
    results
)


# --------------------------------------------------
# SELECT IMPORTANT COLUMNS
# --------------------------------------------------

columns = [
    "model",
    "task",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "training_time_seconds"
]


df = df[
    [
        column
        for column in columns
        if column in df.columns
    ]
]


# --------------------------------------------------
# ROUND VALUES
# --------------------------------------------------

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]:

    if column in df.columns:

        df[column] = df[
            column
        ].round(4)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print(
    "\n========== BASELINE COMPARISON ==========\n"
)

print(
    df.to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    "results/metrics/baseline_comparison.csv",
    index=False
)

print(
    "\nComparison saved to:"
)

print(
    "results/metrics/baseline_comparison.csv"
)