from __future__ import annotations

import torch


PAIR_FEATURE_COMPONENTS = 4


def build_pair_feature(
    premise_embedding: torch.Tensor,
    hypothesis_embedding: torch.Tensor,
    *,
    expected_embedding_dim: int = 1024,
) -> torch.Tensor:
    """Build concat(u, v, |u - v|, u * v) along the last dimension."""

    _validate_embedding(premise_embedding, expected_embedding_dim, "premise_embedding")
    _validate_embedding(hypothesis_embedding, expected_embedding_dim, "hypothesis_embedding")
    if premise_embedding.shape != hypothesis_embedding.shape:
        raise ValueError(
            "Premise and hypothesis embeddings must have the same shape, "
            f"got {tuple(premise_embedding.shape)} and {tuple(hypothesis_embedding.shape)}."
        )

    return torch.cat(
        [
            premise_embedding,
            hypothesis_embedding,
            torch.abs(premise_embedding - hypothesis_embedding),
            premise_embedding * hypothesis_embedding,
        ],
        dim=-1,
    )


def split_pair_feature(
    feature: torch.Tensor,
    *,
    expected_embedding_dim: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the premise and hypothesis embeddings from a pair feature."""

    if feature.ndim < 1:
        raise ValueError(f"feature must have at least one dimension, got {tuple(feature.shape)}.")
    expected_feature_dim = PAIR_FEATURE_COMPONENTS * expected_embedding_dim
    if feature.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"feature last dimension must be {expected_feature_dim}, got {feature.shape[-1]}."
        )

    premise_embedding = feature[..., :expected_embedding_dim]
    hypothesis_embedding = feature[..., expected_embedding_dim : 2 * expected_embedding_dim]
    return premise_embedding, hypothesis_embedding


def _validate_embedding(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    if tensor.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension, got {tuple(tensor.shape)}.")
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {tensor.shape[-1]}.")
