"""
bilstm_model.py
───────────────
Modeling: Bidirectional LSTM architecture for shot outcome prediction.

Architecture:
  Input: (batch, seq_len=30, n_features=12)
    → BiLSTM layer 1 (hidden=128, bidirectional) → Dropout(0.3)
    → BiLSTM layer 2 (hidden=64,  bidirectional) → Dropout(0.3)
    → Attention pooling (weight each timestep)
    → Linear(256, 64) → ReLU → Dropout(0.2)
    → Linear(64, 1) → Sigmoid
  Output: scalar shot make probability [0, 1]

Why BiLSTM?
  - Forward pass: captures the build-up (knee bend → drive → arm extension)
  - Backward pass: captures the follow-through after release
  - Together they learn the complete biomechanical sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Soft attention over timesteps.
    Learns which frames in the shot sequence matter most for prediction.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
        """
        scores = self.attention(lstm_out)          # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)         # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden_dim)
        return context, weights.squeeze(-1)        # also return weights for viz


class ShotPredictor(nn.Module):
    """
    Bidirectional LSTM with attention for basketball shot outcome prediction.

    Parameters
    ----------
    input_size   : number of biomechanical features per timestep
    hidden_size  : LSTM hidden units per direction
    num_layers   : number of stacked LSTM layers
    dropout      : dropout probability between layers
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Input projection (optional feature mixing)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LayerNorm(input_size * 2),
            nn.ReLU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # × 2 for bidirectional

        # Attention pooling
        self.attention = AttentionPooling(lstm_out_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            return_attention: if True, also return attention weights

        Returns:
            prob: (batch_size, 1) — shot make probability
            (optional) attn_weights: (batch_size, seq_len)
        """
        # Input projection
        x = self.input_proj(x)                     # (B, T, input_size*2)

        # BiLSTM
        lstm_out, _ = self.lstm(x)                 # (B, T, hidden*2)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (B, hidden*2)

        # Classification
        prob = self.classifier(context)            # (B, 1)

        if return_attention:
            return prob, attn_weights
        return prob

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning probabilities (no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    input_size: int = 12,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> ShotPredictor:
    """Factory function to build model from config."""
    model = ShotPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    n_params = count_parameters(model)
    print(f"ShotPredictor | Parameters: {n_params:,}")
    print(f"  input_size  : {input_size}")
    print(f"  hidden_size : {hidden_size}")
    print(f"  num_layers  : {num_layers}")
    print(f"  dropout     : {dropout}")
    return model


if __name__ == "__main__":
    # Quick smoke test
    B, T, F = 8, 30, 12
    x = torch.randn(B, T, F)
    model = build_model(input_size=F)
    out, attn = model(x, return_attention=True)
    print(f"\nInput : {x.shape}")
    print(f"Output: {out.shape}  (values in [{out.min():.3f}, {out.max():.3f}])")
    print(f"Attn  : {attn.shape}")
    assert out.shape == (B, 1), "Output shape mismatch"
    print("✅ Smoke test passed")
