import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


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
# SETTINGS
# --------------------------------------------------

BATCH_SIZE = 256

EPOCHS = 10

LEARNING_RATE = 0.001

RANDOM_STATE = 42


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nUsing device:",
    device
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print(
    "\n========== LOADING DATA =========="
)

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


# --------------------------------------------------
# CREATE DATASET
# --------------------------------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


y_train_tensor = torch.tensor(
    y_multiclass_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_multiclass_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==================================================
# CNN-LSTM MODEL
# ==================================================

class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_features=10,
        num_classes=10
    ):

        super().__init__()


        # ------------------------------------------
        # CNN
        # ------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.batch_norm = nn.BatchNorm1d(
            16
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )


        # ------------------------------------------
        # LSTM
        # ------------------------------------------

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=32,
            batch_first=True
        )


        # ------------------------------------------
        # CLASSIFIER
        # ------------------------------------------

        self.fc = nn.Linear(
            32,
            num_classes
        )


    def forward(self, x):

        # Input:
        # [batch, 10]

        # Add channel dimension
        x = x.unsqueeze(1)

        # [batch, 1, 10]

        x = self.conv1(x)

        x = self.batch_norm(x)

        x = self.relu(x)

        x = self.pool(x)

        # CNN output:
        # [batch, 16, sequence_length]

        # Convert for LSTM:
        # [batch, sequence_length, 16]

        x = x.permute(
            0,
            2,
            1
        )

        x, _ = self.lstm(x)

        # Take final time step

        x = x[:, -1, :]

        x = self.fc(x)

        return x


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = CNNLSTM().to(device)

print(
    "\n========== CNN-LSTM MODEL =========="
)

print(model)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAINING
# ==================================================

print(
    "\n========== TRAINING CNN-LSTM =========="
)

start_time = time.time()


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(
            X_batch
        )


        # Calculate loss

        loss = criterion(
            outputs,
            y_batch
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        total_loss += (
            loss.item()
        )


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS} "
        f"- Loss: "
        f"{average_loss:.4f}"
    )


training_time = (
    time.time() - start_time
)

print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


# ==================================================
# TESTING
# ==================================================

print(
    "\n========== TESTING CNN-LSTM =========="
)

model.eval()

all_predictions = []

all_targets = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(
            X_batch
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.numpy()
        )


y_pred = np.array(
    all_predictions
)

y_true = np.array(
    all_targets
)


# ==================================================
# EVALUATION
# ==================================================

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


metrics = calculate_metrics(
    y_true,
    y_pred,
    average="weighted"
)


print_metrics(
    "CNN-LSTM",
    metrics
)


save_metrics(
    "cnn_lstm",
    "multiclass",
    metrics,
    training_time
)


save_confusion_matrix(
    y_true,
    y_pred,
    "cnn_lstm",
    "multiclass",
    class_names
)


print_classification_report(
    y_true,
    y_pred,
    class_names
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "models/baselines",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/baselines/cnn_lstm_multiclass.pth"
)


print(
    "\nCNN-LSTM model saved."
)