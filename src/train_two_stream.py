"""
Training script for the Two-Stream CNN + Attention + BiLSTM model.

The script:
1. Loads the corrected processed dataset
2. Creates DataLoaders
3. Trains the Two-Stream model
4. Validates after every epoch
5. Saves the best model
6. Evaluates on the untouched test set
7. Saves metrics, predictions and plots
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.two_stream_model import TwoStreamModel


# ==========================================================
# 1. Configuration
# ==========================================================

RANDOM_STATE = 42

BATCH_SIZE = 256
LEARNING_RATE = 0.001

EPOCHS = 20
PATIENCE = 4

DROPOUT = 0.20

NUM_CLASSES = 10

MODEL_PATH = Path("models/two_stream_best.pth")

METRICS_DIR = Path("results/metrics")
PREDICTIONS_DIR = Path("results/predictions")
FIGURES_DIR = Path("results/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. Reproducibility
# ==========================================================

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================
# 3. Select device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print("Using:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU.")


# ==========================================================
# 4. Dataset class
# ==========================================================

class TwoStreamDataset(Dataset):

    def __init__(
        self,
        network_features,
        dns_features,
        binary_labels,
        multiclass_labels
    ):

        self.network_features = torch.tensor(
            network_features,
            dtype=torch.float32
        )

        self.dns_features = torch.tensor(
            dns_features,
            dtype=torch.float32
        )

        self.binary_labels = torch.tensor(
            binary_labels,
            dtype=torch.float32
        )

        self.multiclass_labels = torch.tensor(
            multiclass_labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.binary_labels)

    def __getitem__(self, index):

        return (
            self.network_features[index],
            self.dns_features[index],
            self.binary_labels[index],
            self.multiclass_labels[index]
        )


# ==========================================================
# 5. Load processed data
# ==========================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

network_train = np.load(
    "data/processed/final_network_train.npy"
)

dns_train = np.load(
    "data/processed/final_dns_train.npy"
)

binary_train = np.load(
    "data/processed/final_binary_train.npy"
)

multiclass_train = np.load(
    "data/processed/final_multiclass_train.npy"
)


network_val = np.load(
    "data/processed/final_network_val.npy"
)

dns_val = np.load(
    "data/processed/final_dns_val.npy"
)

binary_val = np.load(
    "data/processed/final_binary_val.npy"
)

multiclass_val = np.load(
    "data/processed/final_multiclass_val.npy"
)


network_test = np.load(
    "data/processed/final_network_test.npy"
)

dns_test = np.load(
    "data/processed/final_dns_test.npy"
)

binary_test = np.load(
    "data/processed/final_binary_test.npy"
)

multiclass_test = np.load(
    "data/processed/final_multiclass_test.npy"
)


print("\nTraining data:")
print("Network:", network_train.shape)
print("DNS:", dns_train.shape)
print("Binary:", binary_train.shape)
print("Multiclass:", multiclass_train.shape)

print("\nValidation data:")
print("Network:", network_val.shape)
print("DNS:", dns_val.shape)

print("\nTest data:")
print("Network:", network_test.shape)
print("DNS:", dns_test.shape)


# ==========================================================
# 6. Create datasets
# ==========================================================

train_dataset = TwoStreamDataset(
    network_train,
    dns_train,
    binary_train,
    multiclass_train
)

val_dataset = TwoStreamDataset(
    network_val,
    dns_val,
    binary_val,
    multiclass_val
)

test_dataset = TwoStreamDataset(
    network_test,
    dns_test,
    binary_test,
    multiclass_test
)


# ==========================================================
# 7. Create DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================================
# 8. Create model
# ==========================================================

print("\n" + "=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = TwoStreamModel(
    num_network_features=5,
    num_dns_features=5,
    num_classes=NUM_CLASSES,
    hidden_size=64,
    num_attention_heads=4,
    dropout=DROPOUT
)

model = model.to(device)

print(model)


# ==========================================================
# 9. Loss functions
# ==========================================================

binary_loss_function = nn.BCEWithLogitsLoss()

multiclass_loss_function = nn.CrossEntropyLoss()


# ==========================================================
# 10. Optimizer
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ==========================================================
# 11. Training history
# ==========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_binary_accuracy": [],
    "val_binary_accuracy": [],
    "train_multiclass_accuracy": [],
    "val_multiclass_accuracy": []
}


# ==========================================================
# 12. Training function
# ==========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    binary_predictions = []
    binary_actual = []

    multiclass_predictions = []
    multiclass_actual = []

    for (
        network_x,
        dns_x,
        binary_y,
        multiclass_y
    ) in train_loader:

        network_x = network_x.to(device)
        dns_x = dns_x.to(device)

        binary_y = binary_y.to(device)
        multiclass_y = multiclass_y.to(device)

        optimizer.zero_grad()

        binary_logits, multiclass_logits, _ = model(
            network_x,
            dns_x
        )

        binary_loss = binary_loss_function(
            binary_logits,
            binary_y
        )

        multiclass_loss = multiclass_loss_function(
            multiclass_logits,
            multiclass_y
        )

        # Equal importance to both tasks
        loss = (
            0.5 * binary_loss
            + 0.5 * multiclass_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Binary prediction
        binary_probabilities = torch.sigmoid(
            binary_logits
        )

        binary_pred = (
            binary_probabilities >= 0.5
        ).long()

        binary_predictions.extend(
            binary_pred.cpu().numpy()
        )

        binary_actual.extend(
            binary_y.long().cpu().numpy()
        )

        # Multiclass prediction
        multiclass_pred = torch.argmax(
            multiclass_logits,
            dim=1
        )

        multiclass_predictions.extend(
            multiclass_pred.cpu().numpy()
        )

        multiclass_actual.extend(
            multiclass_y.cpu().numpy()
        )

    average_loss = (
        total_loss / len(train_loader)
    )

    binary_accuracy = accuracy_score(
        binary_actual,
        binary_predictions
    )

    multiclass_accuracy = accuracy_score(
        multiclass_actual,
        multiclass_predictions
    )

    return (
        average_loss,
        binary_accuracy,
        multiclass_accuracy
    )


# ==========================================================
# 13. Validation function
# ==========================================================

def validate():

    model.eval()

    total_loss = 0.0

    binary_predictions = []
    binary_actual = []

    multiclass_predictions = []
    multiclass_actual = []

    with torch.no_grad():

        for (
            network_x,
            dns_x,
            binary_y,
            multiclass_y
        ) in val_loader:

            network_x = network_x.to(device)
            dns_x = dns_x.to(device)

            binary_y = binary_y.to(device)
            multiclass_y = multiclass_y.to(device)

            binary_logits, multiclass_logits, _ = model(
                network_x,
                dns_x
            )

            binary_loss = binary_loss_function(
                binary_logits,
                binary_y
            )

            multiclass_loss = multiclass_loss_function(
                multiclass_logits,
                multiclass_y
            )

            loss = (
                0.5 * binary_loss
                + 0.5 * multiclass_loss
            )

            total_loss += loss.item()

            # Binary
            binary_probability = torch.sigmoid(
                binary_logits
            )

            binary_pred = (
                binary_probability >= 0.5
            ).long()

            binary_predictions.extend(
                binary_pred.cpu().numpy()
            )

            binary_actual.extend(
                binary_y.long().cpu().numpy()
            )

            # Multiclass
            multiclass_pred = torch.argmax(
                multiclass_logits,
                dim=1
            )

            multiclass_predictions.extend(
                multiclass_pred.cpu().numpy()
            )

            multiclass_actual.extend(
                multiclass_y.cpu().numpy()
            )

    average_loss = (
        total_loss / len(val_loader)
    )

    binary_accuracy = accuracy_score(
        binary_actual,
        binary_predictions
    )

    multiclass_accuracy = accuracy_score(
        multiclass_actual,
        multiclass_predictions
    )

    return (
        average_loss,
        binary_accuracy,
        multiclass_accuracy
    )


# ==========================================================
# 14. Training loop
# ==========================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_val_loss = float("inf")

epochs_without_improvement = 0

start_time = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    train_loss, train_binary_acc, train_multi_acc = (
        train_one_epoch()
    )

    val_loss, val_binary_acc, val_multi_acc = (
        validate()
    )

    # Save history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_binary_accuracy"].append(
        train_binary_acc
    )

    history["val_binary_accuracy"].append(
        val_binary_acc
    )

    history["train_multiclass_accuracy"].append(
        train_multi_acc
    )

    history["val_multiclass_accuracy"].append(
        val_multi_acc
    )

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Train Binary Accuracy: "
        f"{train_binary_acc:.4f}"
    )

    print(
        f"Validation Binary Accuracy: "
        f"{val_binary_acc:.4f}"
    )

    print(
        f"Train Multiclass Accuracy: "
        f"{train_multi_acc:.4f}"
    )

    print(
        f"Validation Multiclass Accuracy: "
        f"{val_multi_acc:.4f}"
    )

    print(
        f"Epoch Time: {epoch_time:.2f} seconds"
    )

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


total_training_time = (
    time.time() - start_time
)

print(
    f"\nTotal training time: "
    f"{total_training_time / 60:.2f} minutes"
)


# ==========================================================
# 15. Save training history
# ==========================================================

with open(
    METRICS_DIR / "two_stream_history.json",
    "w"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ==========================================================
# 16. Plot training loss
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Two-Stream Model Training and Validation Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "two_stream_loss.png",
    dpi=300
)

plt.close()


# ==========================================================
# 17. Plot accuracy
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_multiclass_accuracy"],
    label="Training Multiclass Accuracy"
)

plt.plot(
    history["val_multiclass_accuracy"],
    label="Validation Multiclass Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Two-Stream Multiclass Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "two_stream_multiclass_accuracy.png",
    dpi=300
)

plt.close()


# ==========================================================
# 18. Load best model
# ==========================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ==========================================================
# 19. Test evaluation
# ==========================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

binary_predictions = []
binary_probabilities = []
binary_actual = []

multiclass_predictions = []
multiclass_probabilities = []
multiclass_actual = []

attention_outputs = []


with torch.no_grad():

    for (
        network_x,
        dns_x,
        binary_y,
        multiclass_y
    ) in test_loader:

        network_x = network_x.to(device)
        dns_x = dns_x.to(device)

        binary_logits, multiclass_logits, attention = model(
            network_x,
            dns_x
        )

        # Binary probabilities
        binary_probability = torch.sigmoid(
            binary_logits
        )

        binary_pred = (
            binary_probability >= 0.5
        ).long()

        binary_probabilities.extend(
            binary_probability.cpu().numpy()
        )

        binary_predictions.extend(
            binary_pred.cpu().numpy()
        )

        binary_actual.extend(
            binary_y.numpy()
        )

        # Multiclass probabilities
        multiclass_probability = torch.softmax(
            multiclass_logits,
            dim=1
        )

        multiclass_pred = torch.argmax(
            multiclass_logits,
            dim=1
        )

        multiclass_probabilities.extend(
            multiclass_probability.cpu().numpy()
        )

        multiclass_predictions.extend(
            multiclass_pred.cpu().numpy()
        )

        multiclass_actual.extend(
            multiclass_y.numpy()
        )

        attention_outputs.append(
            attention.cpu().numpy()
        )


# Convert lists to arrays

binary_predictions = np.array(
    binary_predictions
)

binary_probabilities = np.array(
    binary_probabilities
)

binary_actual = np.array(
    binary_actual
)

multiclass_predictions = np.array(
    multiclass_predictions
)

multiclass_probabilities = np.array(
    multiclass_probabilities
)

multiclass_actual = np.array(
    multiclass_actual
)

attention_outputs = np.concatenate(
    attention_outputs,
    axis=0
)


# ==========================================================
# 20. Binary metrics
# ==========================================================

binary_accuracy = accuracy_score(
    binary_actual,
    binary_predictions
)

binary_precision = precision_score(
    binary_actual,
    binary_predictions,
    zero_division=0
)

binary_recall = recall_score(
    binary_actual,
    binary_predictions,
    zero_division=0
)

binary_f1 = f1_score(
    binary_actual,
    binary_predictions,
    zero_division=0
)


# ==========================================================
# 21. Multiclass metrics
# ==========================================================

multiclass_accuracy = accuracy_score(
    multiclass_actual,
    multiclass_predictions
)

multiclass_precision = precision_score(
    multiclass_actual,
    multiclass_predictions,
    average="weighted",
    zero_division=0
)

multiclass_recall = recall_score(
    multiclass_actual,
    multiclass_predictions,
    average="weighted",
    zero_division=0
)

multiclass_f1 = f1_score(
    multiclass_actual,
    multiclass_predictions,
    average="weighted",
    zero_division=0
)


# ==========================================================
# 22. Print results
# ==========================================================

print("\nBINARY CLASSIFICATION")
print("-" * 40)

print(
    f"Accuracy : {binary_accuracy:.4f}"
)

print(
    f"Precision: {binary_precision:.4f}"
)

print(
    f"Recall   : {binary_recall:.4f}"
)

print(
    f"F1 Score : {binary_f1:.4f}"
)


print("\nMULTICLASS CLASSIFICATION")
print("-" * 40)

print(
    f"Accuracy : {multiclass_accuracy:.4f}"
)

print(
    f"Precision: {multiclass_precision:.4f}"
)

print(
    f"Recall   : {multiclass_recall:.4f}"
)

print(
    f"F1 Score : {multiclass_f1:.4f}"
)


# ==========================================================
# 23. Classification report
# ==========================================================

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

report = classification_report(
    multiclass_actual,
    multiclass_predictions,
    target_names=class_names,
    zero_division=0
)

print("\nMULTICLASS CLASSIFICATION REPORT")
print("=" * 60)
print(report)


# ==========================================================
# 24. Save metrics
# ==========================================================

metrics = {

    "binary": {
        "accuracy": float(binary_accuracy),
        "precision": float(binary_precision),
        "recall": float(binary_recall),
        "f1": float(binary_f1)
    },

    "multiclass": {
        "accuracy": float(multiclass_accuracy),
        "precision_weighted": float(multiclass_precision),
        "recall_weighted": float(multiclass_recall),
        "f1_weighted": float(multiclass_f1)
    },

    "training_time_minutes": (
        float(total_training_time / 60)
    )
}


with open(
    METRICS_DIR / "two_stream_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================================
# 25. Save classification report
# ==========================================================

with open(
    METRICS_DIR / "two_stream_classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================================
# 26. Save confusion matrices
# ==========================================================

binary_cm = confusion_matrix(
    binary_actual,
    binary_predictions
)

multiclass_cm = confusion_matrix(
    multiclass_actual,
    multiclass_predictions
)


np.save(
    PREDICTIONS_DIR / "two_stream_binary_confusion_matrix.npy",
    binary_cm
)

np.save(
    PREDICTIONS_DIR / "two_stream_multiclass_confusion_matrix.npy",
    multiclass_cm
)


# ==========================================================
# 27. Save predictions
# ==========================================================

np.save(
    PREDICTIONS_DIR / "two_stream_binary_predictions.npy",
    binary_predictions
)

np.save(
    PREDICTIONS_DIR / "two_stream_binary_probabilities.npy",
    binary_probabilities
)

np.save(
    PREDICTIONS_DIR / "two_stream_binary_actual.npy",
    binary_actual
)

np.save(
    PREDICTIONS_DIR / "two_stream_multiclass_predictions.npy",
    multiclass_predictions
)

np.save(
    PREDICTIONS_DIR / "two_stream_multiclass_probabilities.npy",
    multiclass_probabilities
)

np.save(
    PREDICTIONS_DIR / "two_stream_multiclass_actual.npy",
    multiclass_actual
)

np.save(
    PREDICTIONS_DIR / "two_stream_attention.npy",
    attention_outputs
)


# ==========================================================
# 28. Final message
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 60)

print("\nSaved files:")

print(
    "Best model:",
    MODEL_PATH
)

print(
    "Metrics:",
    METRICS_DIR / "two_stream_metrics.json"
)

print(
    "Classification report:",
    METRICS_DIR /
    "two_stream_classification_report.txt"
)

print(
    "Training history:",
    METRICS_DIR /
    "two_stream_history.json"
)

print(
    "Predictions:",
    PREDICTIONS_DIR
)

print(
    "Figures:",
    FIGURES_DIR
)