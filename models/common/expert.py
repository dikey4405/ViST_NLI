from __future__ import annotations

import torch
from torch import nn


class InputProjection(nn.Module):
    """Shared input projection from 4096-dim feature vectors to hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FeedForwardExpert(nn.Module):
    """Shared expert FFN implementation used by all MoE variants."""

    def __init__(self, hidden_dim: int, expert_ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NLIClassifier(nn.Module):
    """Shared NLI classifier head."""

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class ModeNLIHead(NLIClassifier):
    """Shared NLI evidence head applied identically to every modality mode."""

    def forward(self, mode_embeddings: torch.Tensor) -> torch.Tensor:
        if mode_embeddings.ndim != 3:
            raise ValueError(
                "mode_embeddings must have shape [B, M, H], "
                f"got {tuple(mode_embeddings.shape)}."
            )
        return super().forward(mode_embeddings)
