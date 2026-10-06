from pathlib import Path
import pandas as pd


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

TRAIN_PATH = Path("data/raw/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/raw/UNSW_NB15_testing-set.csv")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n========== LOADING DATA ==========")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset loaded.")
print("Testing dataset loaded.")


# --------------------------------------------------
# BASIC SHAPE
# --------------------------------------------------

print("\n========== DATASET SHAPE ==========")

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)


# --------------------------------------------------
# COLUMN NAMES
# --------------------------------------------------

print("\n========== COLUMNS ==========")

for i, column in enumerate(train_df.columns):
    print(f"{i + 1}. {column}")


# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

print("\n========== DATA TYPES ==========")

print(train_df.dtypes)


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = train_df.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n========== DUPLICATES ==========")

print(
    "Training duplicates:",
    train_df.duplicated().sum()
)

print(
    "Testing duplicates:",
    test_df.duplicated().sum()
)


# --------------------------------------------------
# TARGET COLUMNS
# --------------------------------------------------

print("\n========== TARGET INFORMATION ==========")

if "label" in train_df.columns:

    print("\nBinary label distribution:")
    print(
        train_df["label"].value_counts()
    )

if "attack_cat" in train_df.columns:

    print("\nAttack category distribution:")
    print(
        train_df["attack_cat"].value_counts()
    )


# --------------------------------------------------
# CATEGORICAL COLUMNS
# --------------------------------------------------

print("\n========== CATEGORICAL COLUMNS ==========")

categorical_columns = train_df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print(
        f"\n{column}:"
    )

    print(
        train_df[column]
        .value_counts()
        .head(20)
    )


# --------------------------------------------------
# NUMERICAL SUMMARY
# --------------------------------------------------

print("\n========== NUMERICAL SUMMARY ==========")

print(
    train_df.describe().T
)


# --------------------------------------------------
# DATASET COMPARISON
# --------------------------------------------------

print("\n========== TRAIN / TEST COLUMNS ==========")

print(
    "Same columns:",
    list(train_df.columns)
    == list(test_df.columns)
)


print("\n========== INSPECTION COMPLETE ==========")