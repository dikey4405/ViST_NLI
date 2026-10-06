from __future__ import annotations

import torch
from torch import nn


class _ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"embeddings must have shape [B, {self.input_dim}], got {tuple(embeddings.shape)}."
            )
        projected = self.net(embeddings)
        if projected.shape[-1] != self.output_dim:
            raise RuntimeError(f"Projector produced unexpected shape: {tuple(projected.shape)}.")
        return projected


class TextProjector(_ModalityProjector):
    """Trainable projector for pooled PhoBERT embeddings."""


class SpeechProjector(_ModalityProjector):
    """Trainable projector for pooled XLS-R embeddings."""


def build_projector_pair(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    device: torch.device | None = None,
) -> tuple[TextProjector, SpeechProjector]:
    """Build independent text and speech projectors from one architecture contract."""

    text_projector = TextProjector(input_dim, hidden_dim, output_dim, dropout)
    speech_projector = SpeechProjector(input_dim, hidden_dim, output_dim, dropout)
    if device is not None:
        text_projector.to(device)
        speech_projector.to(device)
    return text_projector, speech_projector
