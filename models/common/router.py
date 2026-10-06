from __future__ import annotations

import torch
from torch import nn

from .reliability import ReliabilityEmbedding


class TopKRouter(nn.Module):
    """Softmax router with normalized top-k expert weights."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._route(self.router(x))

    def _route(self, router_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }


class ReliabilityAwareRouter(TopKRouter):
    """Condition top-k routing on semantic, mode, and reliability signals."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        mode_embedding_dim: int,
        reliability_embedding_dim: int,
    ) -> None:
        super().__init__(hidden_dim, num_experts, top_k)
        if mode_embedding_dim < 1:
            raise ValueError(
                f"mode_embedding_dim must be positive, got {mode_embedding_dim}."
            )
        self.mode_embedding_dim = mode_embedding_dim
        self.reliability_embedding = ReliabilityEmbedding(reliability_embedding_dim)
        self.condition_router = nn.Linear(
            mode_embedding_dim + reliability_embedding_dim,
            num_experts,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mode_embeddings: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mode_embeddings is None and reliability is None:
            return super().forward(x)
        if mode_embeddings is None or reliability is None:
            raise ValueError(
                "mode_embeddings and reliability must either both be provided or both be omitted."
            )
        if mode_embeddings.shape != (x.shape[0], self.mode_embedding_dim):
            raise ValueError(
                f"mode_embeddings must have shape {(x.shape[0], self.mode_embedding_dim)}, "
                f"got {tuple(mode_embeddings.shape)}."
            )
        if reliability.shape != (x.shape[0],):
            raise ValueError(
                f"reliability must have shape {(x.shape[0],)}, "
                f"got {tuple(reliability.shape)}."
            )
        if not torch.isfinite(reliability).all():
            raise ValueError("reliability must contain only finite values.")
        if (reliability < 0).any() or (reliability > 1).any():
            raise ValueError("reliability values must be in [0, 1].")

        reliability_features = self.reliability_embedding(reliability.unsqueeze(-1))
        conditioning = torch.cat([mode_embeddings, reliability_features], dim=-1)
        router_logits = self.router(x) + self.condition_router(conditioning)
        return self._route(router_logits)


class CounterfactualUtilityRouter(TopKRouter):
    """Predict expert utility from semantic features and mode identity."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        mode_embedding_dim: int,
    ) -> None:
        super().__init__(hidden_dim, num_experts, top_k)
        if mode_embedding_dim < 1:
            raise ValueError(
                f"mode_embedding_dim must be positive, got {mode_embedding_dim}."
            )
        self.mode_embedding_dim = mode_embedding_dim
        self.mode_utility = nn.Linear(
            mode_embedding_dim,
            num_experts,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mode_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if mode_embeddings.shape != (x.shape[0], self.mode_embedding_dim):
            raise ValueError(
                f"mode_embeddings must have shape {(x.shape[0], self.mode_embedding_dim)}, "
                f"got {tuple(mode_embeddings.shape)}."
            )
        predicted_utility = self.router(x) + self.mode_utility(mode_embeddings)
        return self._route(predicted_utility)
