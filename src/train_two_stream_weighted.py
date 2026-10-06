"""
CyberAgent - Class-Weighted Two-Stream Training

Purpose:
1. Recreate the original train/validation/test split.
2. DO NOT use full SMOTE for deep learning.
3. Use class-weighted CrossEntropyLoss for multiclass classification.
4. Use class-weighted BCEWithLogitsLoss for binary classification.
5. Tune the binary decision threshold using validation data.
6. Save this experiment separately from the original model.

This experiment is designed to reduce:
- Normal -> attack false positives
- Minority-class overprediction
"""

from pathlib import Path
import json
import copy
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.models.two_stream_model import TwoStreamModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_PATH = Path("data/raw/UNSW_NB15_training-set.csv")
TEST_PATH = Path("data/raw/UNSW_NB15_testing-set.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
PRED_DIR = RESULT_DIR / "predictions"
METRIC_DIR = RESULT_DIR / "metrics"

MODEL_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "two_stream_weighted_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Smaller batch and fewer epochs are intentional.
# Your previous training took ~77 minutes on CPU.
BATCH_SIZE = 512
EPOCHS = 12
PATIENCE = 3

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Same loss balance as your previous model
BINARY_LOSS_WEIGHT = 0.5
MULTICLASS_LOSS_WEIGHT = 0.5

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


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_raw_data():

    print("=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training data: {train_df.shape}")
    print(f"Testing data : {test_df.shape}")

    return train_df, test_df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(train_df, test_df):

    """
    Recreates the same 5 network + 5 DNS features
    used by the existing Two-Stream model.

    Important:
    These DNS features are flow-level proxies because
    UNSW-NB15 CSV data does not contain raw DNS query content.
    """

    train = train_df.copy()
    test = test_df.copy()

    # --------------------------------------------------------
    # Combine only for consistent categorical encoding.
    # Feature selection/scaling are still fitted ONLY on train.
    # --------------------------------------------------------

    combined = pd.concat(
        [
            train.drop(columns=["attack_cat", "label"], errors="ignore"),
            test.drop(columns=["attack_cat", "label"], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )

    categorical_columns = ["proto", "service", "state"]

    combined = pd.get_dummies(
        combined,
        columns=categorical_columns,
        dummy_na=False,
    )

    # Restore train/test portions
    train_encoded = combined.iloc[: len(train)].copy()
    test_encoded = combined.iloc[len(train):].copy()

    # --------------------------------------------------------
    # Helper for safe numeric conversion
    # --------------------------------------------------------

    def numeric(df, column):

        if column not in df.columns:
            return pd.Series(
                np.zeros(len(df)),
                index=df.index,
                dtype=float,
            )

        return pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # Network features
    # --------------------------------------------------------

    def build_network_features(df):

        spkts = numeric(df, "spkts")
        dpkts = numeric(df, "dpkts")
        sbytes = numeric(df, "sbytes")
        dbytes = numeric(df, "dbytes")

        total_packets = spkts + dpkts
        total_packets = total_packets.replace(0, 1)

        packet_rate = numeric(df, "rate")

        flow_duration = numeric(df, "dur")

        packet_size = (
            (sbytes + dbytes) / total_packets
        )

        inter_arrival_time = (
            numeric(df, "sinpkt") +
            numeric(df, "dinpkt")
        ) / 2.0

        # Same proxy idea as original preprocessing:
        # identify the protocol with the highest one-hot value.
        proto_columns = [
            c for c in df.columns
            if c.startswith("proto_")
        ]

        if proto_columns:

            protocol_distribution = (
                df[proto_columns]
                .astype(float)
                .idxmax(axis=1)
                .astype("category")
                .cat.codes
            )

        else:

            protocol_distribution = pd.Series(
                np.zeros(len(df)),
                index=df.index,
            )

        result = pd.DataFrame(
            {
                "packet_rate": packet_rate,
                "flow_duration": flow_duration,
                "packet_size": packet_size,
                "inter_arrival_time": inter_arrival_time,
                "protocol_distribution":
                    protocol_distribution,
            },
            index=df.index,
        )

        return result.replace(
            [np.inf, -np.inf],
            0,
        ).fillna(0)

    # --------------------------------------------------------
    # DNS proxy features
    # --------------------------------------------------------

    def build_dns_features(df):

        spkts = numeric(df, "spkts")
        dpkts = numeric(df, "dpkts")
        sbytes = numeric(df, "sbytes")
        dbytes = numeric(df, "dbytes")

        # DNS service indicator
        dns_indicator = pd.Series(
            0.0,
            index=df.index,
        )

        dns_columns = [
            c for c in df.columns
            if c.lower() == "service_dns"
        ]

        if dns_columns:
            dns_indicator = numeric(
                df,
                dns_columns[0],
            )

        query_rate = (
            numeric(df, "rate") *
            dns_indicator
        )

        safe_spkts = spkts.replace(0, 1)

        query_length = (
            sbytes / safe_spkts
        ) * dns_indicator

        dns_query_frequency = (
            (spkts + dpkts) *
            dns_indicator
        )

        query_type_distribution = dns_indicator

        dns_response_activity = (
            dbytes *
            dns_indicator
        )

        result = pd.DataFrame(
            {
                "query_rate": query_rate,
                "query_length": query_length,
                "dns_query_frequency":
                    dns_query_frequency,
                "query_type_distribution":
                    query_type_distribution,
                "dns_response_activity":
                    dns_response_activity,
            },
            index=df.index,
        )

        return result.replace(
            [np.inf, -np.inf],
            0,
        ).fillna(0)

    network_train = build_network_features(train_encoded)
    network_test = build_network_features(test_encoded)

    dns_train = build_dns_features(train_encoded)
    dns_test = build_dns_features(test_encoded)

    return (
        network_train,
        network_test,
        dns_train,
        dns_test,
    )


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

def split_data(
    network_train,
    network_test,
    dns_train,
    dns_test,
    train_df,
):

    print("\n" + "=" * 70)
    print("CREATING TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    y_binary = train_df["label"].astype(int).values

    attack_categories = train_df["attack_cat"].astype(str)

    class_to_id = {
        name: idx
        for idx, name in enumerate(CLASS_NAMES)
    }

    y_multiclass = attack_categories.map(
        class_to_id
    ).values

    indices = np.arange(len(train_df))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y_multiclass,
    )

    print(f"Training samples  : {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples      : {len(network_test)}")

    return (
        train_indices,
        val_indices,
        y_binary,
        y_multiclass,
    )


# ============================================================
# 6. RIDGE FEATURE SELECTION
# ============================================================

def select_and_scale_features(
    network_train,
    network_test,
    dns_train,
    dns_test,
    train_indices,
    val_indices,
    y_multiclass,
):

    print("\n" + "=" * 70)
    print("RIDGE FEATURE SELECTION + SCALING")
    print("=" * 70)

    # --------------------------------------------------------
    # NETWORK
    # --------------------------------------------------------

    network_scaler = StandardScaler()

    network_train_scaled = network_scaler.fit_transform(
        network_train.iloc[train_indices]
    )

    network_val_scaled = network_scaler.transform(
        network_train.iloc[val_indices]
    )

    network_test_scaled = network_scaler.transform(
        network_test
    )

    network_ridge = Ridge(
        alpha=1.0
    )

    network_ridge.fit(
        network_train_scaled,
        y_multiclass[train_indices],
    )

    network_scores = np.abs(
        network_ridge.coef_
    )

    network_ranking = pd.DataFrame(
        {
            "feature": network_train.columns,
            "importance": network_scores,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    selected_network = (
        network_ranking
        .head(5)["feature"]
        .tolist()
    )

    print("\nSelected Network Features:")
    for feature in selected_network:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # DNS
    # --------------------------------------------------------

    dns_scaler = StandardScaler()

    dns_train_scaled = dns_scaler.fit_transform(
        dns_train.iloc[train_indices]
    )

    dns_val_scaled = dns_scaler.transform(
        dns_train.iloc[val_indices]
    )

    dns_test_scaled = dns_scaler.transform(
        dns_test
    )

    dns_ridge = Ridge(
        alpha=1.0
    )

    dns_ridge.fit(
        dns_train_scaled,
        y_multiclass[train_indices],
    )

    dns_scores = np.abs(
        dns_ridge.coef_
    )

    dns_ranking = pd.DataFrame(
        {
            "feature": dns_train.columns,
            "importance": dns_scores,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    selected_dns = (
        dns_ranking
        .head(5)["feature"]
        .tolist()
    )

    print("\nSelected DNS Features:")
    for feature in selected_dns:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Return only selected features
    # --------------------------------------------------------

    network_train_final = network_train_scaled[
        :,
        [
            network_train.columns.get_loc(x)
            for x in selected_network
        ],
    ]

    network_val_final = network_val_scaled[
        :,
        [
            network_train.columns.get_loc(x)
            for x in selected_network
        ],
    ]

    network_test_final = network_test_scaled[
        :,
        [
            network_train.columns.get_loc(x)
            for x in selected_network
        ],
    ]

    dns_train_final = dns_train_scaled[
        :,
        [
            dns_train.columns.get_loc(x)
            for x in selected_dns
        ],
    ]

    dns_val_final = dns_val_scaled[
        :,
        [
            dns_train.columns.get_loc(x)
            for x in selected_dns
        ],
    ]

    dns_test_final = dns_test_scaled[
        :,
        [
            dns_train.columns.get_loc(x)
            for x in selected_dns
        ],
    ]

    return (
        network_train_final,
        network_val_final,
        network_test_final,
        dns_train_final,
        dns_val_final,
        dns_test_final,
        selected_network,
        selected_dns,
    )


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y):

    print("\n" + "=" * 70)
    print("CALCULATING MULTICLASS CLASS WEIGHTS")
    print("=" * 70)

    counts = np.bincount(
        y,
        minlength=len(CLASS_NAMES),
    )

    total = len(y)
    num_classes = len(CLASS_NAMES)

    weights = total / (
        num_classes * np.maximum(counts, 1)
    )

    # Normalize around 1.
    weights = weights / weights.mean()

    print("\nClass distribution and weights:")

    for i, name in enumerate(CLASS_NAMES):

        print(
            f"{name:18s} "
            f"count={counts[i]:7d} "
            f"weight={weights[i]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# 8. DATASET CREATION
# ============================================================

def create_loaders(
    network_train,
    network_val,
    network_test,
    dns_train,
    dns_val,
    dns_test,
    y_binary_train,
    y_binary_val,
    y_binary_test,
    y_multi_train,
    y_multi_val,
    y_multi_test,
):

    train_dataset = TensorDataset(
        torch.tensor(
            network_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            dns_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_binary_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_multi_train,
            dtype=torch.long,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            network_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            dns_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_binary_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_multi_val,
            dtype=torch.long,
        ),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            network_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            dns_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_binary_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_multi_test,
            dtype=torch.long,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader,
    binary_threshold=0.5,
):

    model.eval()

    binary_actual = []
    binary_probabilities = []

    multiclass_actual = []
    multiclass_predictions = []

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for (
            network,
            dns,
            y_binary,
            y_multi,
        ) in loader:

            network = network.to(DEVICE)
            dns = dns.to(DEVICE)
            y_binary = y_binary.to(DEVICE)
            y_multi = y_multi.to(DEVICE)

            binary_logits, multi_logits, _ = model(
                network,
                dns,
            )

            binary_probability = torch.sigmoid(
                binary_logits
            )

            binary_prediction = (
                binary_probability >= binary_threshold
            ).long()

            multi_prediction = torch.argmax(
                multi_logits,
                dim=1,
            )

            binary_actual.extend(
                y_binary.cpu().numpy().astype(int)
            )

            binary_probabilities.extend(
                binary_probability.cpu().numpy()
            )

            multiclass_actual.extend(
                y_multi.cpu().numpy()
            )

            multiclass_predictions.extend(
                multi_prediction.cpu().numpy()
            )

            batches += 1

    binary_actual = np.array(binary_actual)
    binary_probabilities = np.array(binary_probabilities)

    binary_predictions = (
        binary_probabilities >= binary_threshold
    ).astype(int)

    multiclass_actual = np.array(
        multiclass_actual
    )

    multiclass_predictions = np.array(
        multiclass_predictions
    )

    return {
        "binary_actual": binary_actual,
        "binary_probabilities": binary_probabilities,
        "binary_predictions": binary_predictions,
        "multiclass_actual": multiclass_actual,
        "multiclass_predictions": multiclass_predictions,
    }


# ============================================================
# 10. BINARY THRESHOLD SEARCH
# ============================================================

def find_best_binary_threshold(
    actual,
    probabilities,
):

    print("\n" + "=" * 70)
    print("SEARCHING FOR BEST BINARY THRESHOLD")
    print("=" * 70)

    results = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.02,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            actual,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print(
        f"\nBest validation threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Validation precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best['f1']:.4f}"
    )

    results_df.to_csv(
        METRIC_DIR /
        "weighted_binary_threshold_search.csv",
        index=False,
    )

    return float(best["threshold"])


# ============================================================
# 11. TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    multiclass_weights,
    binary_pos_weight,
):

    print("\n" + "=" * 70)
    print("STARTING CLASS-WEIGHTED TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Multiclass weighted loss
    # --------------------------------------------------------

    multiclass_loss_fn = nn.CrossEntropyLoss(
        weight=multiclass_weights
    )

    # --------------------------------------------------------
    # Binary weighted loss
    # --------------------------------------------------------

    binary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=binary_pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        total_train_loss = 0.0
        train_binary_correct = 0
        train_multi_correct = 0
        train_samples = 0

        for (
            network,
            dns,
            y_binary,
            y_multi,
        ) in train_loader:

            network = network.to(DEVICE)
            dns = dns.to(DEVICE)
            y_binary = y_binary.to(DEVICE)
            y_multi = y_multi.to(DEVICE)

            optimizer.zero_grad()

            binary_logits, multi_logits, _ = model(
                network,
                dns,
            )

            binary_loss = binary_loss_fn(
                binary_logits,
                y_binary,
            )

            multiclass_loss = multiclass_loss_fn(
                multi_logits,
                y_multi,
            )

            loss = (
                BINARY_LOSS_WEIGHT * binary_loss
                +
                MULTICLASS_LOSS_WEIGHT * multiclass_loss
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            binary_predictions = (
                torch.sigmoid(
                    binary_logits
                ) >= 0.5
            ).long()

            multiclass_predictions = torch.argmax(
                multi_logits,
                dim=1,
            )

            train_binary_correct += (
                binary_predictions == y_binary.long()
            ).sum().item()

            train_multi_correct += (
                multiclass_predictions == y_multi
            ).sum().item()

            train_samples += len(y_multi)

        train_loss = (
            total_train_loss /
            len(train_loader)
        )

        train_binary_acc = (
            train_binary_correct /
            train_samples
        )

        train_multi_acc = (
            train_multi_correct /
            train_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0.0
        val_samples = 0

        val_binary_correct = 0
        val_multi_correct = 0

        with torch.no_grad():

            for (
                network,
                dns,
                y_binary,
                y_multi,
            ) in val_loader:

                network = network.to(DEVICE)
                dns = dns.to(DEVICE)
                y_binary = y_binary.to(DEVICE)
                y_multi = y_multi.to(DEVICE)

                binary_logits, multi_logits, _ = model(
                    network,
                    dns,
                )

                binary_loss = binary_loss_fn(
                    binary_logits,
                    y_binary,
                )

                multiclass_loss = multiclass_loss_fn(
                    multi_logits,
                    y_multi,
                )

                loss = (
                    BINARY_LOSS_WEIGHT * binary_loss
                    +
                    MULTICLASS_LOSS_WEIGHT * multiclass_loss
                )

                total_val_loss += loss.item()

                binary_predictions = (
                    torch.sigmoid(
                        binary_logits
                    ) >= 0.5
                ).long()

                multiclass_predictions = torch.argmax(
                    multi_logits,
                    dim=1,
                )

                val_binary_correct += (
                    binary_predictions ==
                    y_binary.long()
                ).sum().item()

                val_multi_correct += (
                    multiclass_predictions ==
                    y_multi
                ).sum().item()

                val_samples += len(y_multi)

        val_loss = (
            total_val_loss /
            len(val_loader)
        )

        val_binary_acc = (
            val_binary_correct /
            val_samples
        )

        val_multi_acc = (
            val_multi_correct /
            val_samples
        )

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Binary: {val_binary_acc:.4f} | "
            f"Multi: {val_multi_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_binary_accuracy":
                    train_binary_acc,
                "train_multiclass_accuracy":
                    train_multi_acc,
                "val_binary_accuracy":
                    val_binary_acc,
                "val_multiclass_accuracy":
                    val_multi_acc,
            }
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_val_loss": best_val_loss,
                },
                MODEL_PATH,
            )

            print(
                f"  -> Saved best model to "
                f"{MODEL_PATH}"
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    pd.DataFrame(history).to_csv(
        METRIC_DIR /
        "weighted_training_history.csv",
        index=False,
    )

    model.load_state_dict(best_state)

    return model


# ============================================================
# 12. FINAL EVALUATION
# ============================================================

def print_final_results(
    results,
    threshold,
):

    actual_binary = results[
        "binary_actual"
    ]

    pred_binary = (
        results["binary_probabilities"]
        >= threshold
    ).astype(int)

    actual_multi = results[
        "multiclass_actual"
    ]

    pred_multi = results[
        "multiclass_predictions"
    ]

    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL BINARY RESULTS")
    print("=" * 70)

    binary_accuracy = accuracy_score(
        actual_binary,
        pred_binary,
    )

    binary_precision = precision_score(
        actual_binary,
        pred_binary,
        zero_division=0,
    )

    binary_recall = recall_score(
        actual_binary,
        pred_binary,
        zero_division=0,
    )

    binary_f1 = f1_score(
        actual_binary,
        pred_binary,
        zero_division=0,
    )

    cm_binary = confusion_matrix(
        actual_binary,
        pred_binary,
    )

    tn, fp, fn, tp = cm_binary.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {binary_accuracy:.4f}")
    print(f"Precision : {binary_precision:.4f}")
    print(f"Recall    : {binary_recall:.4f}")
    print(f"F1        : {binary_f1:.4f}")
    print(f"FPR       : {fpr:.4f}")
    print(f"FNR       : {fnr:.4f}")

    # --------------------------------------------------------
    # Multiclass
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MULTICLASS RESULTS")
    print("=" * 70)

    multi_accuracy = accuracy_score(
        actual_multi,
        pred_multi,
    )

    multi_precision = precision_score(
        actual_multi,
        pred_multi,
        average="weighted",
        zero_division=0,
    )

    multi_recall = recall_score(
        actual_multi,
        pred_multi,
        average="weighted",
        zero_division=0,
    )

    multi_f1 = f1_score(
        actual_multi,
        pred_multi,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        actual_multi,
        pred_multi,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy     : {multi_accuracy:.4f}")
    print(f"Precision    : {multi_precision:.4f}")
    print(f"Recall       : {multi_recall:.4f}")
    print(f"Weighted F1  : {multi_f1:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            actual_multi,
            pred_multi,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "binary": {
            "threshold": threshold,
            "accuracy": float(binary_accuracy),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "multiclass": {
            "accuracy": float(multi_accuracy),
            "precision_weighted":
                float(multi_precision),
            "recall_weighted":
                float(multi_recall),
            "f1_weighted":
                float(multi_f1),
            "f1_macro":
                float(macro_f1),
        },
    }

    with open(
        METRIC_DIR /
        "weighted_two_stream_metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.save(
        PRED_DIR /
        "weighted_binary_actual.npy",
        actual_binary,
    )

    np.save(
        PRED_DIR /
        "weighted_binary_probabilities.npy",
        results["binary_probabilities"],
    )

    np.save(
        PRED_DIR /
        "weighted_binary_predictions.npy",
        pred_binary,
    )

    np.save(
        PRED_DIR /
        "weighted_multiclass_actual.npy",
        actual_multi,
    )

    np.save(
        PRED_DIR /
        "weighted_multiclass_predictions.npy",
        pred_multi,
    )

    print("\nSaved weighted experiment results.")


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CYBERAGENT - CLASS-WEIGHTED TWO-STREAM MODEL")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df, test_df = load_raw_data()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    (
        network_train,
        network_test,
        dns_train,
        dns_test,
    ) = create_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        y_binary,
        y_multiclass,
    ) = split_data(
        network_train,
        network_test,
        dns_train,
        dns_test,
        train_df,
    )

    # --------------------------------------------------------
    # Select + scale
    # --------------------------------------------------------

    (
        network_train_final,
        network_val_final,
        network_test_final,
        dns_train_final,
        dns_val_final,
        dns_test_final,
        selected_network,
        selected_dns,
    ) = select_and_scale_features(
        network_train,
        network_test,
        dns_train,
        dns_test,
        train_indices,
        val_indices,
        y_multiclass,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_binary_train = y_binary[
        train_indices
    ]

    y_binary_val = y_binary[
        val_indices
    ]

    y_binary_test = test_df[
        "label"
    ].astype(int).values

    y_multi_train = y_multiclass[
        train_indices
    ]

    y_multi_val = y_multiclass[
        val_indices
    ]

    y_multi_test = test_df[
        "attack_cat"
    ].astype(str).map(
        {
            name: idx
            for idx, name in enumerate(CLASS_NAMES)
        }
    ).values

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    multiclass_weights = calculate_class_weights(
        y_multi_train
    )

    # Binary positive weight
    normal_count = np.sum(
        y_binary_train == 0
    )

    attack_count = np.sum(
        y_binary_train == 1
    )

    pos_weight_value = (
        normal_count /
        max(attack_count, 1)
    )

    binary_pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        f"\nBinary positive weight: "
        f"{pos_weight_value:.4f}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders(
        network_train_final,
        network_val_final,
        network_test_final,
        dns_train_final,
        dns_val_final,
        dns_test_final,
        y_binary_train,
        y_binary_val,
        y_binary_test,
        y_multi_train,
        y_multi_val,
        y_multi_test,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TwoStreamModel(
    num_network_features=5,
    num_dns_features=5,
    num_classes=10,
    hidden_size=64,
    num_attention_heads=4,
    dropout=0.2,
).to(DEVICE)

    print("\nModel created.")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        model,
        train_loader,
        val_loader,
        multiclass_weights,
        binary_pos_weight,
    )

    # --------------------------------------------------------
    # Validation threshold tuning
    # --------------------------------------------------------

    val_results = evaluate_model(
        model,
        val_loader,
        binary_threshold=0.5,
    )

    best_threshold = find_best_binary_threshold(
        val_results["binary_actual"],
        val_results["binary_probabilities"],
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        test_loader,
        binary_threshold=best_threshold,
    )

    print_final_results(
        test_results,
        best_threshold,
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": "class_weighted_two_stream",
        "smote": False,
        "architecture": "TwoStreamModel",
        "network_features": selected_network,
        "dns_features": selected_dns,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "binary_threshold": best_threshold,
        "binary_pos_weight": pos_weight_value,
        "class_names": CLASS_NAMES,
    }

    with open(
        METRIC_DIR /
        "weighted_two_stream_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("WEIGHTED EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nModel saved at:\n"
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()