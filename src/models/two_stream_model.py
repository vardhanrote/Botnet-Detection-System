"""
Two-Stream CNN + Multi-Head Attention + BiLSTM model.

The model has two separate input streams:

1. Network traffic features
2. DNS-related features

The two streams are processed separately using CNN layers.
Their representations are then fused and passed through
Multi-Head Attention and a BiLSTM.

The model produces:
- Binary prediction: Normal vs Attack
- Multiclass prediction: 10 traffic/attack categories
"""

import torch
import torch.nn as nn


class FeatureBranch(nn.Module):
    """
    CNN branch used for either network or DNS features.
    """

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # First convolution
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Reduce feature sequence length
            nn.MaxPool1d(kernel_size=2),

            # Second convolution
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input:
            x -> [batch_size, 5]

        Output:
            [batch_size, 2, 32]
        """

        # Add channel dimension
        # [B, 5] -> [B, 1, 5]
        x = x.unsqueeze(1)

        # CNN processing
        x = self.cnn(x)

        # CNN output:
        # [B, 32, 2]

        # Convert to sequence format:
        # [B, 2, 32]
        x = x.transpose(1, 2)

        return x


class TwoStreamModel(nn.Module):
    """
    Main Two-Stream model.

    Network features and DNS features are processed separately,
    then fused using Multi-Head Attention and BiLSTM.
    """

    def __init__(
        self,
        num_network_features=5,
        num_dns_features=5,
        num_classes=10,
        hidden_size=64,
        num_attention_heads=4,
        dropout=0.20
    ):
        super().__init__()

        # Separate CNN branches
        self.network_branch = FeatureBranch()
        self.dns_branch = FeatureBranch()

        # Each branch produces 32 channels.
        # After fusion:
        # 32 + 32 = 64
        fusion_dim = 64

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Final dense layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Binary classification head
        self.binary_head = nn.Linear(128, 1)

        # 10-class classification head
        self.multiclass_head = nn.Linear(128, num_classes)

    def forward(self, network_x, dns_x):
        """
        Parameters
        ----------
        network_x:
            Network features [B, 5]

        dns_x:
            DNS features [B, 5]

        Returns
        -------
        binary_logits:
            [B]

        multiclass_logits:
            [B, 10]

        attention_weights:
            Attention information for explainability
        """

        # --------------------------------------------------
        # 1. Process network features
        # --------------------------------------------------

        network_features = self.network_branch(network_x)

        # Shape:
        # [B, 2, 32]

        # --------------------------------------------------
        # 2. Process DNS features
        # --------------------------------------------------

        dns_features = self.dns_branch(dns_x)

        # Shape:
        # [B, 2, 32]

        # --------------------------------------------------
        # 3. Feature fusion
        # --------------------------------------------------

        # Concatenate feature representations
        fused = torch.cat(
            [network_features, dns_features],
            dim=2
        )

        # Shape:
        # [B, 2, 64]

        # --------------------------------------------------
        # 4. Multi-Head Attention
        # --------------------------------------------------

        attention_output, attention_weights = self.attention(
            fused,
            fused,
            fused
        )

        # Residual connection
        fused = fused + attention_output

        # --------------------------------------------------
        # 5. BiLSTM
        # --------------------------------------------------

        lstm_output, _ = self.bilstm(fused)

        # Shape:
        # [B, 2, 128]
        #
        # 128 = 64 forward + 64 backward

        # --------------------------------------------------
        # 6. Mean pooling
        # --------------------------------------------------

        representation = lstm_output.mean(dim=1)

        # Shape:
        # [B, 128]

        # --------------------------------------------------
        # 7. Dense representation
        # --------------------------------------------------

        representation = self.classifier(representation)

        # --------------------------------------------------
        # 8. Binary prediction
        # --------------------------------------------------

        binary_logits = self.binary_head(
            representation
        ).squeeze(1)

        # --------------------------------------------------
        # 9. Multiclass prediction
        # --------------------------------------------------

        multiclass_logits = self.multiclass_head(
            representation
        )

        return (
            binary_logits,
            multiclass_logits,
            attention_weights
        )


# ----------------------------------------------------------
# Simple model test
# ----------------------------------------------------------

if __name__ == "__main__":

    print("Testing TwoStreamModel...")

    # Create model
    model = TwoStreamModel()

    # Create fake network input
    network_input = torch.randn(4, 5)

    # Create fake DNS input
    dns_input = torch.randn(4, 5)

    # Run model
    binary_output, multiclass_output, attention = model(
        network_input,
        dns_input
    )

    print(
        "Network input shape:",
        network_input.shape
    )

    print(
        "DNS input shape:",
        dns_input.shape
    )

    print(
        "Binary output shape:",
        binary_output.shape
    )

    print(
        "Multiclass output shape:",
        multiclass_output.shape
    )

    print(
        "Attention shape:",
        attention.shape
    )

    print("\nModel test successful.")