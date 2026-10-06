"""
Two-Stream CNN + Multi-Head Attention + BiLSTM model

The model has:
1. Network feature branch
2. DNS feature branch
3. Feature fusion
4. Multi-Head Self Attention
5. Residual connection
6. BiLSTM
7. Dense layer
8. Binary classification head
9. 10-class attack classification head
"""

import torch
import torch.nn as nn


class FeatureBranch(nn.Module):
    """
    Processes one feature stream.

    Example:
    Network stream -> Conv1D -> BatchNorm -> ReLU -> MaxPool -> Conv1D
    DNS stream     -> Conv1D -> BatchNorm -> ReLU -> MaxPool -> Conv1D
    """

    def __init__(self, input_features=5):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(16)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        """
        Input shape:
        [batch_size, 5]

        Conv1D expects:
        [batch_size, channels, sequence_length]

        Therefore we convert:

        [B, 5]
            ↓
        [B, 1, 5]
        """

        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Change from:
        # [B, 32, sequence_length]
        #
        # to:
        # [B, sequence_length, 32]

        x = x.transpose(1, 2)

        return x


class TwoStreamModel(nn.Module):
    """
    Complete two-stream cybersecurity attack detection model.

    Inputs:
        network_features -> 5 features
        dns_features     -> 5 features

    Outputs:
        binary_logits     -> normal vs attack
        multiclass_logits -> 10 attack categories
    """

    def __init__(
        self,
        network_features=5,
        dns_features=5,
        attention_heads=4,
        lstm_hidden=64,
        dropout=0.20,
        num_classes=10
    ):
        super().__init__()

        # ----------------------------------------
        # 1. Network branch
        # ----------------------------------------

        self.network_branch = FeatureBranch(
            input_features=network_features
        )

        # ----------------------------------------
        # 2. DNS branch
        # ----------------------------------------

        self.dns_branch = FeatureBranch(
            input_features=dns_features
        )

        # ----------------------------------------
        # 3. Feature Fusion
        # ----------------------------------------
        #
        # Each branch produces:
        # [B, sequence_length, 32]
        #
        # Concatenating channels gives:
        # [B, sequence_length, 64]

        self.fusion_size = 64

        # ----------------------------------------
        # 4. Multi-Head Attention
        # ----------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_size,
            num_heads=attention_heads,
            batch_first=True
        )

        # ----------------------------------------
        # 5. BiLSTM
        # ----------------------------------------

        self.bilstm = nn.LSTM(
            input_size=self.fusion_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # Because BiLSTM is bidirectional:
        # output size = hidden_size * 2

        lstm_output_size = lstm_hidden * 2

        # ----------------------------------------
        # 6. Dense layer
        # ----------------------------------------

        self.fc1 = nn.Linear(
            lstm_output_size,
            128
        )

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        # ----------------------------------------
        # 7. Binary classification head
        # ----------------------------------------

        self.binary_head = nn.Linear(
            128,
            1
        )

        # ----------------------------------------
        # 8. Multiclass classification head
        # ----------------------------------------

        self.multiclass_head = nn.Linear(
            128,
            num_classes
        )

    def forward(self, network_x, dns_x):

        # ----------------------------------------
        # Network stream
        # ----------------------------------------

        network_features = self.network_branch(
            network_x
        )

        # ----------------------------------------
        # DNS stream
        # ----------------------------------------

        dns_features = self.dns_branch(
            dns_x
        )

        # ----------------------------------------
        # Feature fusion
        # ----------------------------------------

        fused = torch.cat(
            [
                network_features,
                dns_features
            ],
            dim=2
        )

        # fused shape:
        # [B, sequence_length, 64]

        # ----------------------------------------
        # Multi-Head Attention
        # ----------------------------------------

        attention_output, attention_weights = self.attention(
            fused,
            fused,
            fused
        )

        # ----------------------------------------
        # Residual connection
        # ----------------------------------------

        attention_output = attention_output + fused

        # ----------------------------------------
        # BiLSTM
        # ----------------------------------------

        lstm_output, _ = self.bilstm(
            attention_output
        )

        # ----------------------------------------
        # Global average pooling
        # ----------------------------------------

        pooled = torch.mean(
            lstm_output,
            dim=1
        )

        # ----------------------------------------
        # Dense layer
        # ----------------------------------------

        x = self.fc1(pooled)

        x = self.relu(x)

        x = self.dropout(x)

        # ----------------------------------------
        # Binary output
        # ----------------------------------------

        binary_logits = self.binary_head(x)

        binary_logits = binary_logits.squeeze(1)

        # ----------------------------------------
        # Multiclass output
        # ----------------------------------------

        multiclass_logits = self.multiclass_head(x)

        return (
            binary_logits,
            multiclass_logits,
            attention_weights
        )


if __name__ == "__main__":

    # Simple model test

    model = TwoStreamModel()

    print(model)

    # Create dummy data

    network = torch.randn(4, 5)

    dns = torch.randn(4, 5)

    # Forward pass

    binary_output, multiclass_output, attention = model(
        network,
        dns
    )

    print("\nNetwork input:")
    print(network.shape)

    print("\nDNS input:")
    print(dns.shape)

    print("\nBinary output:")
    print(binary_output.shape)

    print("\nMulticlass output:")
    print(multiclass_output.shape)

    print("\nAttention output:")
    print(attention.shape)