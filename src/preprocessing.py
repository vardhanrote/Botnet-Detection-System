from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from imblearn.over_sampling import SMOTE


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TRAIN_PATH = Path("data/raw/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/raw/UNSW_NB15_testing-set.csv")

PROCESSED_DIR = Path("data/processed")


# ============================================================
# TARGET COLUMNS
# ============================================================

BINARY_TARGET = "label"
MULTICLASS_TARGET = "attack_cat"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the original UNSW-NB15 training and testing datasets.
    """

    print("\n========== LOADING DATA ==========")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Training shape:", train_df.shape)
    print("Testing shape :", test_df.shape)

    return train_df, test_df


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_data(df):
    """
    Perform basic cleaning.

    The UNSW-NB15 files currently have no missing values,
    but this function keeps the pipeline robust.
    """

    df = df.copy()

    # Remove duplicate rows
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed = before - len(df)

    print(f"Duplicates removed: {removed}")

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numeric missing values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    return df


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_categorical_features(
    train_df,
    test_df
):
    """
    Convert categorical network fields into numeric values.

    We use one-hot encoding for:
        proto
        service
        state
    """

    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    train_df = train_df.copy()
    test_df = test_df.copy()

    combined = pd.concat(
        [train_df, test_df],
        axis=0
    )

    combined = pd.get_dummies(
        combined,
        columns=categorical_columns,
        dtype=int
    )

    train_encoded = combined.iloc[
        :len(train_df)
    ].copy()

    test_encoded = combined.iloc[
        len(train_df):
    ].copy()

    print("\nCategorical features encoded.")

    return train_encoded, test_encoded


# ============================================================
# CREATE NETWORK FEATURES
# ============================================================

def create_network_features(df):
    """
    Create network-flow features from the available
    UNSW-NB15 columns.

    These represent:
        packet rate
        flow duration
        packet size
        inter-arrival behaviour
        protocol behaviour
    """

    result = pd.DataFrame(index=df.index)

    # 1. Packet rate
    result["packet_rate"] = df["rate"]

    # 2. Flow duration
    result["flow_duration"] = df["dur"]

    # 3. Average packet size
    total_packets = (
        df["spkts"] +
        df["dpkts"]
    )

    total_bytes = (
        df["sbytes"] +
        df["dbytes"]
    )

    result["packet_size"] = (
        total_bytes /
        total_packets.replace(0, np.nan)
    )

    # 4. Inter-arrival time
    result["inter_arrival_time"] = (
        df["sinpkt"] +
        df["dinpkt"]
    ) / 2

    # 5. Protocol distribution proxy
    # The original dataset provides the protocol
    # as a categorical flow-level field.
    #
    # We represent protocol using the encoded
    # protocol columns where available.

    protocol_columns = [
        column
        for column in df.columns
        if column.startswith("proto_")
    ]

    if protocol_columns:

        # A simple numeric protocol representation
        # that keeps the information available to
        # the model.
        result["protocol_distribution"] = (
            df[protocol_columns]
            .idxmax(axis=1)
            .astype("category")
            .cat.codes
        )

    else:

        result["protocol_distribution"] = 0

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.fillna(0)

    return result


# ============================================================
# CREATE DNS-FLOW FEATURES
# ============================================================

def create_dns_features(df):
    """
    Create DNS-flow proxy features.

    IMPORTANT:
    The standard UNSW-NB15 training/testing CSV does not
    contain raw DNS query fields such as domain names,
    query type, or query entropy.

    Therefore these are flow-level DNS proxy features,
    not raw DNS query measurements.
    """

    result = pd.DataFrame(index=df.index)

    # DNS service indicator
    if "service_dns" in df.columns:

        dns_indicator = df["service_dns"]

    else:

        dns_indicator = pd.Series(
            0,
            index=df.index
        )

    # 1. Query-rate proxy
    result["query_rate"] = (
        df["rate"] * dns_indicator
    )

    # 2. Query-length proxy
    # Use source/destination byte behaviour
    # because actual DNS query length is unavailable.
    result["query_length"] = (
        df["sbytes"] /
        df["spkts"].replace(0, np.nan)
    )

    # 3. DNS frequency proxy
    result["dns_query_frequency"] = (
        df["spkts"] +
        df["dpkts"]
    ) * dns_indicator

    # 4. Query-type distribution proxy
    # UNSW-NB15 does not provide DNS query types.
    # We therefore use the service indicator.
    result["query_type_distribution"] = dns_indicator

    # 5. DNS activity / response proxy
    result["dns_response_activity"] = (
        df["dbytes"] *
        dns_indicator
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.fillna(0)

    return result


# ============================================================
# BUILD FEATURE SET
# ============================================================

def build_feature_sets(train_df, test_df):

    # ---------------------------------------------
    # Encode categorical columns first
    # ---------------------------------------------

    train_encoded, test_encoded = encode_categorical_features(
        train_df,
        test_df
    )

    # ---------------------------------------------
    # Create network features
    # ---------------------------------------------

    network_train = create_network_features(
        train_encoded
    )

    network_test = create_network_features(
        test_encoded
    )

    # ---------------------------------------------
    # Create DNS features
    # ---------------------------------------------

    dns_train = create_dns_features(
        train_encoded
    )

    dns_test = create_dns_features(
        test_encoded
    )

    print(
        "Network features:",
        list(network_train.columns)
    )

    print(
        "DNS features:",
        list(dns_train.columns)
    )

    return (
        network_train,
        network_test,
        dns_train,
        dns_test
    )


# ============================================================
# RIDGE FEATURE SELECTION
# ============================================================

def ridge_selection(
    X,
    y,
    number_of_features
):
    """
    Rank features using Ridge Regression.

    IMPORTANT:
    Feature selection is fitted on the training data only.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_scaled,
        y
    )

    importance = np.abs(
        ridge.coef_
    )

    ranking = pd.DataFrame({
        "feature": X.columns,
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


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    train_df,
    test_df
):
    """
    Standardize features.

    The scaler is fitted only on training data.
    """

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_df
    )

    test_scaled = scaler.transform(
        test_df
    )

    train_scaled = pd.DataFrame(
        train_scaled,
        columns=train_df.columns,
        index=train_df.index
    )

    test_scaled = pd.DataFrame(
        test_scaled,
        columns=test_df.columns,
        index=test_df.index
    )

    return (
        train_scaled,
        test_scaled,
        scaler
    )


# ============================================================
# SMOTE
# ============================================================

def apply_smote(
    X_train,
    y_train
):
    """
    Apply SMOTE only to training data.

    This prevents information from the test set
    from entering the training process.
    """

    print("\n========== APPLYING SMOTE ==========")

    print(
        "Before SMOTE:"
    )

    print(
        y_train.value_counts()
    )

    smote = SMOTE(
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print(
        "\nAfter SMOTE:"
    )

    print(
        pd.Series(y_resampled).value_counts()
    )

    return (
        X_resampled,
        y_resampled
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_processed_data(
    network_train,
    network_test,
    dns_train,
    dns_test,
    y_train_binary,
    y_test_binary,
    y_train_multiclass,
    y_test_multiclass
):
    """
    Save the processed feature sets.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    network_train.to_csv(
        PROCESSED_DIR /
        "network_train.csv",
        index=False
    )

    network_test.to_csv(
        PROCESSED_DIR /
        "network_test.csv",
        index=False
    )

    dns_train.to_csv(
        PROCESSED_DIR /
        "dns_train.csv",
        index=False
    )

    dns_test.to_csv(
        PROCESSED_DIR /
        "dns_test.csv",
        index=False
    )

    pd.DataFrame({
        "label": y_train_binary
    }).to_csv(
        PROCESSED_DIR /
        "binary_train.csv",
        index=False
    )

    pd.DataFrame({
        "label": y_test_binary
    }).to_csv(
        PROCESSED_DIR /
        "binary_test.csv",
        index=False
    )

    pd.DataFrame({
        "attack_cat": y_train_multiclass
    }).to_csv(
        PROCESSED_DIR /
        "multiclass_train.csv",
        index=False
    )

    pd.DataFrame({
        "attack_cat": y_test_multiclass
    }).to_csv(
        PROCESSED_DIR /
        "multiclass_test.csv",
        index=False
    )

    print(
        "\nProcessed datasets saved successfully."
    )