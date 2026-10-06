# ============================================================
# FINAL DATA PREPARATION FOR TWO-STREAM MODEL
# ============================================================
#
# Pipeline:
# UNSW-NB15
#      ↓
# Cleaning
#      ↓
# Feature engineering
#      ↓
# Train / Validation split
#      ↓
# Ridge feature selection
#      ↓
# Scaling
#      ↓
# SMOTE ONLY ON TRAINING DATA
#      ↓
# Save Train / Validation / Test
#
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from imblearn.over_sampling import SMOTE

from src.preprocessing import (
    load_data,
    clean_data,
    build_feature_sets
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10

PROCESSED_DIR = Path("data/processed")

NETWORK_FEATURE_COUNT = 5
DNS_FEATURE_COUNT = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: LOADING ORIGINAL DATA")
print("=" * 60)

train_df, test_df = load_data()

print("Training data:", train_df.shape)
print("Testing data :", test_df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: CLEANING DATA")
print("=" * 60)

train_df = clean_data(train_df)
test_df = clean_data(test_df)


# ============================================================
# 3. SAVE ORIGINAL LABELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: PREPARING LABELS")
print("=" * 60)

# Binary labels
y_binary = train_df["label"].astype(int).values
y_test_binary = test_df["label"].astype(int).values


# ============================================================
# 4. CONVERT ATTACK CATEGORIES TO NUMERIC IDs
# ============================================================

# Fixed class order.
# This is important because the neural network's output
# position must always correspond to the same attack class.

class_names = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"
]

class_to_id = {
    name: index
    for index, name in enumerate(class_names)
}


def encode_attack_categories(series):
    """
    Convert attack category names into integer IDs.
    """

    encoded = series.map(class_to_id)

    if encoded.isnull().any():

        unknown = series[encoded.isnull()].unique()

        raise ValueError(
            f"Unknown attack categories found: {unknown}"
        )

    return encoded.astype(int).values


y_multiclass = encode_attack_categories(
    train_df["attack_cat"]
)

y_test_multiclass = encode_attack_categories(
    test_df["attack_cat"]
)


print("\nClass mapping:")

for name, index in class_to_id.items():
    print(f"{index}: {name}")


# ============================================================
# 5. CREATE NETWORK + DNS FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: CREATING NETWORK AND DNS FEATURES")
print("=" * 60)

(
    network_train_full,
    network_test,
    dns_train_full,
    dns_test
) = build_feature_sets(
    train_df,
    test_df
)


print("\nNetwork features:")
print(list(network_train_full.columns))

print("\nDNS features:")
print(list(dns_train_full.columns))


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================
#
# IMPORTANT:
#
# We split BEFORE:
# - Ridge feature selection
# - scaling
# - SMOTE
#
# This prevents validation leakage.
#
# Stratification is done using the 10-class target because
# some attack categories are highly imbalanced.
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: TRAIN / VALIDATION SPLIT")
print("=" * 60)

indices = np.arange(len(train_df))

train_indices, validation_indices = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_multiclass
)


print("Training samples  :", len(train_indices))
print("Validation samples:", len(validation_indices))
print("Testing samples   :", len(test_df))


# ============================================================
# 7. SPLIT FEATURES
# ============================================================

network_train_split = network_train_full.iloc[
    train_indices
].reset_index(drop=True)

network_validation = network_train_full.iloc[
    validation_indices
].reset_index(drop=True)

dns_train_split = dns_train_full.iloc[
    train_indices
].reset_index(drop=True)

dns_validation = dns_train_full.iloc[
    validation_indices
].reset_index(drop=True)


# ============================================================
# SPLIT LABELS
# ============================================================

y_train_binary = y_binary[
    train_indices
]

y_validation_binary = y_binary[
    validation_indices
]

y_train_multiclass = y_multiclass[
    train_indices
]

y_validation_multiclass = y_multiclass[
    validation_indices
]


# ============================================================
# 8. RIDGE FEATURE SELECTION
# ============================================================
#
# Ridge is fitted ONLY on the training split.
#
# We select:
#   5 network features
#   5 DNS features
#
# This preserves the two-stream architecture.
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: RIDGE FEATURE SELECTION")
print("=" * 60)


def ridge_select(
    X_train,
    y_train,
    number_of_features
):
    """
    Perform Ridge-based feature ranking.

    The model is fitted only on training data.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_scaled,
        y_train
    )

    importance = np.abs(
        ridge.coef_
    )

    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    })

    ranking = ranking.sort_values(
        "importance",
        ascending=False
    )

    selected = ranking.head(
        number_of_features
    )["feature"].tolist()

    return selected, ranking


# ------------------------------------------------------------
# Network feature selection
# ------------------------------------------------------------

selected_network_features, network_ranking = ridge_select(
    network_train_split,
    y_train_multiclass,
    NETWORK_FEATURE_COUNT
)


# ------------------------------------------------------------
# DNS feature selection
# ------------------------------------------------------------

selected_dns_features, dns_ranking = ridge_select(
    dns_train_split,
    y_train_multiclass,
    DNS_FEATURE_COUNT
)


print("\nSelected network features:")

for feature in selected_network_features:
    print("-", feature)


print("\nSelected DNS features:")

for feature in selected_dns_features:
    print("-", feature)


# ============================================================
# 9. KEEP ONLY SELECTED FEATURES
# ============================================================

network_train_split = network_train_split[
    selected_network_features
]

network_validation = network_validation[
    selected_network_features
]

network_test_selected = network_test[
    selected_network_features
]


dns_train_split = dns_train_split[
    selected_dns_features
]

dns_validation = dns_validation[
    selected_dns_features
]

dns_test_selected = dns_test[
    selected_dns_features
]


# ============================================================
# 10. SCALE FEATURES
# ============================================================
#
# The scaler is fitted ONLY on the training split.
#
# Validation and test data are transformed using the
# training scaler.
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: SCALING FEATURES")
print("=" * 60)


# ------------------------------------------------------------
# Network scaler
# ------------------------------------------------------------

network_scaler = StandardScaler()

network_train_scaled = network_scaler.fit_transform(
    network_train_split
)

network_validation_scaled = network_scaler.transform(
    network_validation
)

network_test_scaled = network_scaler.transform(
    network_test_selected
)


# ------------------------------------------------------------
# DNS scaler
# ------------------------------------------------------------

dns_scaler = StandardScaler()

dns_train_scaled = dns_scaler.fit_transform(
    dns_train_split
)

dns_validation_scaled = dns_scaler.transform(
    dns_validation
)

dns_test_scaled = dns_scaler.transform(
    dns_test_selected
)


# ============================================================
# 11. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================
#
# We combine the two feature streams temporarily only for
# SMOTE because SMOTE requires one feature matrix.
#
# After SMOTE, we split the features back into:
#   Network = 5 features
#   DNS     = 5 features
#
# Validation and test are NEVER passed through SMOTE.
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: APPLYING SMOTE")
print("=" * 60)


X_train_combined = np.concatenate(
    [
        network_train_scaled,
        dns_train_scaled
    ],
    axis=1
)


print("\nBefore SMOTE:")

unique, counts = np.unique(
    y_train_multiclass,
    return_counts=True
)

for class_id, count in zip(unique, counts):
    print(
        f"{class_id} - "
        f"{class_names[class_id]}: "
        f"{count}"
    )


smote = SMOTE(
    random_state=RANDOM_STATE
)


X_train_resampled, y_train_multiclass_resampled = (
    smote.fit_resample(
        X_train_combined,
        y_train_multiclass
    )
)


print("\nAfter SMOTE:")

unique, counts = np.unique(
    y_train_multiclass_resampled,
    return_counts=True
)

for class_id, count in zip(unique, counts):
    print(
        f"{class_id} - "
        f"{class_names[class_id]}: "
        f"{count}"
    )


# ============================================================
# 12. SPLIT SMOTE OUTPUT BACK INTO TWO STREAMS
# ============================================================

network_train_final = X_train_resampled[
    :,
    :NETWORK_FEATURE_COUNT
]

dns_train_final = X_train_resampled[
    :,
    NETWORK_FEATURE_COUNT:
]


# ============================================================
# 13. CREATE BINARY LABELS FOR SMOTE DATA
# ============================================================
#
# Class 0 = Normal
# Classes 1-9 = Attacks
#
# Therefore:
# Normal -> 0
# Any attack -> 1
# ============================================================

y_train_binary_resampled = (
    y_train_multiclass_resampled != class_to_id["Normal"]
).astype(int)


# ============================================================
# 14. CONVERT VALIDATION AND TEST DATA TO NUMPY
# ============================================================

network_validation_final = np.asarray(
    network_validation_scaled,
    dtype=np.float32
)

dns_validation_final = np.asarray(
    dns_validation_scaled,
    dtype=np.float32
)

network_test_final = np.asarray(
    network_test_scaled,
    dtype=np.float32
)

dns_test_final = np.asarray(
    dns_test_scaled,
    dtype=np.float32
)


# Training arrays
network_train_final = np.asarray(
    network_train_final,
    dtype=np.float32
)

dns_train_final = np.asarray(
    dns_train_final,
    dtype=np.float32
)

y_train_binary_resampled = np.asarray(
    y_train_binary_resampled,
    dtype=np.int64
)

y_train_multiclass_resampled = np.asarray(
    y_train_multiclass_resampled,
    dtype=np.int64
)


# Validation labels
y_validation_binary = np.asarray(
    y_validation_binary,
    dtype=np.int64
)

y_validation_multiclass = np.asarray(
    y_validation_multiclass,
    dtype=np.int64
)


# Test labels
y_test_binary = np.asarray(
    y_test_binary,
    dtype=np.int64
)

y_test_multiclass = np.asarray(
    y_test_multiclass,
    dtype=np.int64
)


# ============================================================
# 15. SAVE FINAL NUMPY ARRAYS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: SAVING FINAL DATA")
print("=" * 60)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

np.save(
    PROCESSED_DIR / "final_network_train.npy",
    network_train_final
)

np.save(
    PROCESSED_DIR / "final_dns_train.npy",
    dns_train_final
)

np.save(
    PROCESSED_DIR / "final_binary_train.npy",
    y_train_binary_resampled
)

np.save(
    PROCESSED_DIR / "final_multiclass_train.npy",
    y_train_multiclass_resampled
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

np.save(
    PROCESSED_DIR / "final_network_val.npy",
    network_validation_final
)

np.save(
    PROCESSED_DIR / "final_dns_val.npy",
    dns_validation_final
)

np.save(
    PROCESSED_DIR / "final_binary_val.npy",
    y_validation_binary
)

np.save(
    PROCESSED_DIR / "final_multiclass_val.npy",
    y_validation_multiclass
)


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

np.save(
    PROCESSED_DIR / "final_network_test.npy",
    network_test_final
)

np.save(
    PROCESSED_DIR / "final_dns_test.npy",
    dns_test_final
)

np.save(
    PROCESSED_DIR / "final_binary_test.npy",
    y_test_binary
)

np.save(
    PROCESSED_DIR / "final_multiclass_test.npy",
    y_test_multiclass
)


# ============================================================
# 16. SAVE FEATURE INFORMATION
# ============================================================

feature_metadata = {

    "network_features": selected_network_features,

    "dns_features": selected_dns_features,

    "network_feature_count": NETWORK_FEATURE_COUNT,

    "dns_feature_count": DNS_FEATURE_COUNT,

    "class_names": class_names,

    "class_to_id": class_to_id,

    "random_state": RANDOM_STATE,

    "validation_size": VALIDATION_SIZE,

    "smote_applied": True,

    "smote_applied_only_to_training": True,

    "dns_features_are_flow_proxies": True

}


with open(
    PROCESSED_DIR / "feature_metadata.json",
    "w"
) as file:

    json.dump(
        feature_metadata,
        file,
        indent=4
    )


# ============================================================
# 17. SAVE RIDGE RANKINGS
# ============================================================

network_ranking.to_csv(
    PROCESSED_DIR / "network_ridge_ranking.csv",
    index=False
)

dns_ranking.to_csv(
    PROCESSED_DIR / "dns_ridge_ranking.csv",
    index=False
)


# ============================================================
# 18. PRINT FINAL SHAPES
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET SHAPES")
print("=" * 60)

print(
    "\nTraining:"
)

print(
    "Network:",
    network_train_final.shape
)

print(
    "DNS:",
    dns_train_final.shape
)

print(
    "Binary:",
    y_train_binary_resampled.shape
)

print(
    "Multiclass:",
    y_train_multiclass_resampled.shape
)


print(
    "\nValidation:"
)

print(
    "Network:",
    network_validation_final.shape
)

print(
    "DNS:",
    dns_validation_final.shape
)

print(
    "Binary:",
    y_validation_binary.shape
)

print(
    "Multiclass:",
    y_validation_multiclass.shape
)


print(
    "\nTesting:"
)

print(
    "Network:",
    network_test_final.shape
)

print(
    "DNS:",
    dns_test_final.shape
)

print(
    "Binary:",
    y_test_binary.shape
)

print(
    "Multiclass:",
    y_test_multiclass.shape
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA PREPARATION COMPLETE")
print("=" * 60)

print(
    "\nSaved files inside:",
    PROCESSED_DIR
)