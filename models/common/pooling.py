from __future__ import annotations

import torch
from torch import nn


class ModeAttentionPooling(nn.Module):
    """Learn a normalized importance weight for each input mode."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        use_reliability: bool = False,
        reliability_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if use_reliability and reliability_hidden_dim < 1:
            raise ValueError(
                "reliability_hidden_dim must be positive when reliability is enabled, "
                f"got {reliability_hidden_dim}."
            )
        self.hidden_dim = hidden_dim
        self.use_reliability = use_reliability
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.reliability_score = (
            nn.Sequential(
                nn.Linear(1, reliability_hidden_dim),
                nn.Tanh(),
                nn.Linear(reliability_hidden_dim, 1, bias=False),
            )
            if use_reliability
            else None
        )

    def forward(
        self,
        mode_embeddings: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode_embeddings.ndim != 3:
            raise ValueError(
                "mode_embeddings must have shape [B, M, H], "
                f"got {tuple(mode_embeddings.shape)}."
            )
        if mode_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"mode_embeddings last dimension must be {self.hidden_dim}, "
                f"got {mode_embeddings.shape[-1]}."
            )
        if mode_embeddings.shape[1] < 1:
            raise ValueError("mode_embeddings must contain at least one mode.")

        scores = self.score(mode_embeddings).squeeze(-1)
        if self.use_reliability:
            expected_shape = mode_embeddings.shape[:2]
            if reliability is None or reliability.shape != expected_shape:
                actual_shape = None if reliability is None else tuple(reliability.shape)
                raise ValueError(
                    f"reliability must have shape {tuple(expected_shape)}, got {actual_shape}."
                )
            scores = scores + self.reliability_score(reliability.unsqueeze(-1)).squeeze(-1)
        elif reliability is not None:
            raise ValueError("This pooling module was created without reliability conditioning.")
        attention_weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(
            attention_weights.unsqueeze(-1) * mode_embeddings,
            dim=1,
        )
        return pooled, attention_weights
