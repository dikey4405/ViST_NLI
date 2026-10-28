from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


SUPPORTED_EXPERT_ACTIVATIONS = ("relu", "gelu", "silu", "swiglu")


def normalize_expert_activation(activation_function: str) -> str:
    """Return a canonical expert activation name or raise a clear error."""

    if not isinstance(activation_function, str):
        raise ValueError(
            "activation_function must be a string, "
            f"got {type(activation_function).__name__}."
        )
    normalized = activation_function.strip().lower()
    if normalized not in SUPPORTED_EXPERT_ACTIVATIONS:
        supported = ", ".join(SUPPORTED_EXPERT_ACTIVATIONS)
        raise ValueError(
            f"Unsupported activation_function {activation_function!r}. "
            f"Expected one of: {supported}."
        )
    return normalized


def parameter_matched_swiglu_dim(expert_ffn_dim: int) -> int:
    """Return the nearest two-thirds SwiGLU width used for parameter matching."""

    if expert_ffn_dim < 1:
        raise ValueError(
            f"expert_ffn_dim must be positive, got {expert_ffn_dim}."
        )
    return max(1, round(2 * expert_ffn_dim / 3))


class SwiGLU(nn.Module):
    """Parameter-free SwiGLU gate over a fused value/gate projection."""

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.shape[-1] % 2 != 0:
            raise ValueError(
                "SwiGLU input dimension must be even so it can be split into "
                f"value and gate branches, got {projected.shape[-1]}."
            )
        value, gate = projected.chunk(2, dim=-1)
        return value * F.silu(gate)


def _build_expert_activation(activation_function: str) -> nn.Module:
    if activation_function == "relu":
        return nn.ReLU()
    if activation_function == "gelu":
        return nn.GELU()
    if activation_function == "silu":
        return nn.SiLU()
    return SwiGLU()


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

    def __init__(
        self,
        hidden_dim: int,
        expert_ffn_dim: int,
        dropout: float,
        activation_function: str = "gelu",
    ) -> None:
        super().__init__()
        self.activation_function = normalize_expert_activation(activation_function)
        self.expert_ffn_dim = expert_ffn_dim
        self.effective_ffn_dim = (
            parameter_matched_swiglu_dim(expert_ffn_dim)
            if self.activation_function == "swiglu"
            else expert_ffn_dim
        )
        input_projection_dim = (
            2 * self.effective_ffn_dim
            if self.activation_function == "swiglu"
            else self.effective_ffn_dim
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, input_projection_dim),
            _build_expert_activation(self.activation_function),
            nn.Dropout(dropout),
            nn.Linear(self.effective_ffn_dim, hidden_dim),
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
