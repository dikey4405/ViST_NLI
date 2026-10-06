from __future__ import annotations

import torch


def masked_mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool valid sequence positions while excluding padded positions."""

    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [B, T, D], got {tuple(hidden_states.shape)}."
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [B, T], got {tuple(attention_mask.shape)}."
        )
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            "hidden_states and attention_mask sequence dimensions must match, "
            f"got {tuple(hidden_states.shape[:2])} and {tuple(attention_mask.shape)}."
        )

    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    valid_counts = mask.sum(dim=1, keepdim=True)
    if torch.any(valid_counts == 0):
        raise ValueError("Each sequence must contain at least one unmasked position.")
    return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
