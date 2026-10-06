import os
import sys
import time

import numpy as np
import joblib

from xgboost import XGBClassifier


# Allow importing evaluate.py
sys.path.append(
    os.path.dirname(__file__)
)

from evaluate import (
    calculate_metrics,
    print_metrics,
    save_metrics,
    save_confusion_matrix,
    print_classification_report
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n========== LOADING DATA ==========")

X_network_train = np.load(
    "data/processed/X_network_train.npy"
)

X_dns_train = np.load(
    "data/processed/X_dns_train.npy"
)

X_network_test = np.load(
    "data/processed/X_network_test.npy"
)

X_dns_test = np.load(
    "data/processed/X_dns_test.npy"
)

y_binary_train = np.load(
    "data/processed/y_binary_train.npy"
)

y_binary_test = np.load(
    "data/processed/y_binary_test.npy"
)

y_multiclass_train = np.load(
    "data/processed/y_multiclass_train.npy"
)

y_multiclass_test = np.load(
    "data/processed/y_multiclass_test.npy"
)


# --------------------------------------------------
# COMBINE FEATURES
# --------------------------------------------------

X_train = np.concatenate(
    [
        X_network_train,
        X_dns_train
    ],
    axis=1
)

X_test = np.concatenate(
    [
        X_network_test,
        X_dns_test
    ],
    axis=1
)


print(
    "Training shape:",
    X_train.shape
)

print(
    "Testing shape:",
    X_test.shape
)


# ==================================================
# BINARY XGBOOST
# ==================================================

print(
    "\n========== XGBOOST - BINARY =========="
)

start_time = time.time()

binary_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

binary_model.fit(
    X_train,
    y_binary_train
)

training_time = (
    time.time() - start_time
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_binary_pred = binary_model.predict(
    X_test
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

binary_metrics = calculate_metrics(
    y_binary_test,
    y_binary_pred,
    average="binary"
)

print_metrics(
    "XGBoost",
    binary_metrics
)

save_metrics(
    "xgboost",
    "binary",
    binary_metrics,
    training_time
)

save_confusion_matrix(
    y_binary_test,
    y_binary_pred,
    "xgboost",
    "binary",
    class_names=[
        "Normal",
        "Attack"
    ]
)


# ==================================================
# MULTICLASS XGBOOST
# ==================================================

print(
    "\n========== XGBOOST - MULTICLASS =========="
)

start_time = time.time()

multiclass_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=10,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

multiclass_model.fit(
    X_train,
    y_multiclass_train
)

training_time = (
    time.time() - start_time
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

y_multiclass_pred = (
    multiclass_model.predict(
        X_test
    )
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

multiclass_metrics = calculate_metrics(
    y_multiclass_test,
    y_multiclass_pred,
    average="weighted"
)

print_metrics(
    "XGBoost",
    multiclass_metrics
)

save_metrics(
    "xgboost",
    "multiclass",
    multiclass_metrics,
    training_time
)


class_names = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Normal",
    "Reconnaissance",
    "Shellcode",
    "Worms"
]


save_confusion_matrix(
    y_multiclass_test,
    y_multiclass_pred,
    "xgboost",
    "multiclass",
    class_names
)


print_classification_report(
    y_multiclass_test,
    y_multiclass_pred,
    class_names
)


# --------------------------------------------------
# SAVE MODELS
# --------------------------------------------------

os.makedirs(
    "models/baselines",
    exist_ok=True
)

joblib.dump(
    binary_model,
    "models/baselines/xgboost_binary.pkl"
)

joblib.dump(
    multiclass_model,
    "models/baselines/xgboost_multiclass.pkl"
)

print(
    "\nXGBoost models saved."
)