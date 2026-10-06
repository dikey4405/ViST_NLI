from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_pt(payload: Any, path: str | Path) -> None:
    torch.save(payload, Path(path))


def load_pt(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu")


def validate_embedding_shape(
    tensor: torch.Tensor,
    *,
    expected_dim: int = 1024,
    name: str = "embedding",
) -> None:
    _validate_vector_shape(tensor, expected_dim=expected_dim, name=name)


def validate_feature_shape(
    tensor: torch.Tensor,
    *,
    expected_dim: int = 4096,
    name: str = "feature",
) -> None:
    _validate_vector_shape(tensor, expected_dim=expected_dim, name=name)


def _validate_vector_shape(tensor: torch.Tensor, *, expected_dim: int, name: str) -> None:
    if tensor.ndim not in {1, 2}:
        raise ValueError(
            f"{name} must have shape [{expected_dim}] or "
            f"[batch_size, {expected_dim}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} last dimension must be {expected_dim}, got {tuple(tensor.shape)}")


__all__ = [
    "ensure_output_dir",
    "load_pt",
    "save_pt",
    "validate_embedding_shape",
    "validate_feature_shape",
]
