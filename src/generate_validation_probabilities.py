"""
Generate validation-set probabilities for:
1. Random Forest
2. XGBoost
3. Original Two-Stream model

These probabilities will be used to optimize ensemble weights.

IMPORTANT:
The validation set is used for model selection.
The final test set remains untouched until the selected
ensemble weights are frozen.
"""

import os
import joblib
import numpy as np
import torch

from src.models.two_stream_model import TwoStreamModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/baselines"
PREDICTION_DIR = "results/predictions"

os.makedirs(PREDICTION_DIR, exist_ok=True)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("GENERATING VALIDATION MODEL PROBABILITIES")
print("=" * 70)


network_val = np.load(
    os.path.join(PROCESSED_DIR, "final_network_val.npy")
)

dns_val = np.load(
    os.path.join(PROCESSED_DIR, "final_dns_val.npy")
)

y_val = np.load(
    os.path.join(PROCESSED_DIR, "final_multiclass_val.npy")
)

print("Network validation:", network_val.shape)
print("DNS validation    :", dns_val.shape)
print("Labels            :", y_val.shape)


# Combined features for traditional ML models
X_val = np.concatenate(
    [network_val, dns_val],
    axis=1
)

print("Combined validation:", X_val.shape)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nLoading Random Forest...")

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest_multiclass.pkl"
)

rf_model = joblib.load(rf_path)

rf_probabilities = rf_model.predict_proba(X_val)

print(
    "RF probabilities:",
    rf_probabilities.shape
)


# ============================================================
# XGBOOST
# ============================================================

print("\nLoading XGBoost...")

xgb_path = os.path.join(
    MODEL_DIR,
    "xgboost_multiclass.pkl"
)

xgb_model = joblib.load(xgb_path)

xgb_probabilities = xgb_model.predict_proba(X_val)

print(
    "XGBoost probabilities:",
    xgb_probabilities.shape
)


# ============================================================
# TWO-STREAM MODEL
# ============================================================

print("\nLoading Two-Stream model...")

two_stream_path = (
    "models/two_stream_best.pth"
)

model = TwoStreamModel(
    num_network_features=5,
    num_dns_features=5,
    num_classes=10,
    hidden_size=64,
    num_attention_heads=4,
    dropout=0.2,
).to(DEVICE)


checkpoint = torch.load(
    two_stream_path,
    map_location=DEVICE
)


# Handle both possible checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)


model.eval()


# Convert validation data to tensors
network_tensor = torch.tensor(
    network_val,
    dtype=torch.float32
).to(DEVICE)

dns_tensor = torch.tensor(
    dns_val,
    dtype=torch.float32
).to(DEVICE)


# Run inference
all_two_stream_probabilities = []


with torch.no_grad():

    batch_size = 512

    for start in range(
        0,
        len(network_tensor),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(network_tensor)
        )

        network_batch = network_tensor[start:end]
        dns_batch = dns_tensor[start:end]

        binary_logits, multiclass_logits, _ = model(
            network_batch,
            dns_batch
        )

        probabilities = torch.softmax(
            multiclass_logits,
            dim=1
        )

        all_two_stream_probabilities.append(
            probabilities.cpu().numpy()
        )


two_stream_probabilities = np.concatenate(
    all_two_stream_probabilities,
    axis=0
)


print(
    "Two-Stream probabilities:",
    two_stream_probabilities.shape
)


# ============================================================
# VERIFY SHAPES
# ============================================================

assert rf_probabilities.shape == (
    len(y_val),
    10
)

assert xgb_probabilities.shape == (
    len(y_val),
    10
)

assert two_stream_probabilities.shape == (
    len(y_val),
    10
)


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        PREDICTION_DIR,
        "validation_rf_multiclass_probabilities.npy"
    ),
    rf_probabilities
)

np.save(
    os.path.join(
        PREDICTION_DIR,
        "validation_xgboost_multiclass_probabilities.npy"
    ),
    xgb_probabilities
)

np.save(
    os.path.join(
        PREDICTION_DIR,
        "validation_two_stream_multiclass_probabilities.npy"
    ),
    two_stream_probabilities
)

np.save(
    os.path.join(
        PREDICTION_DIR,
        "validation_multiclass_actual.npy"
    ),
    y_val
)


print("\nFiles saved successfully.")

print(
    "\nSaved files:"
)

print(
    "results/predictions/"
    "validation_rf_multiclass_probabilities.npy"
)

print(
    "results/predictions/"
    "validation_xgboost_multiclass_probabilities.npy"
)

print(
    "results/predictions/"
    "validation_two_stream_multiclass_probabilities.npy"
)

print(
    "results/predictions/"
    "validation_multiclass_actual.npy"
)