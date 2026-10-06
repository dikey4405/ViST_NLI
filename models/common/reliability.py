from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EntropyReliabilityEstimator(nn.Module):
    """Convert per-mode class logits to confidence via normalized entropy."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
        self.num_classes = num_classes
        self.max_entropy = math.log(num_classes)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have shape [B, M, C], got {tuple(logits.shape)}."
            )
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"logits last dimension must be {self.num_classes}, "
                f"got {logits.shape[-1]}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values.")

        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        normalized_entropy = -(probabilities * log_probs).sum(dim=-1) / self.max_entropy
        return (1.0 - normalized_entropy).clamp(min=0.0, max=1.0)


class ReliabilityEmbedding(nn.Module):
    """Map one entropy-derived reliability scalar to a small routing feature."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, reliability: torch.Tensor) -> torch.Tensor:
        if reliability.ndim != 2 or reliability.shape[-1] != 1:
            raise ValueError(
                "reliability must have shape [N, 1], "
                f"got {tuple(reliability.shape)}."
            )
        return self.net(reliability)
