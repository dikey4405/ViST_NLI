from __future__ import annotations

from typing import Any

import torch
from torch import nn


def validate_feature_tensor(
    features: torch.Tensor,
    *,
    input_dim: int,
    num_modes: int,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, {num_modes}, {input_dim}], got {tuple(features.shape)}.")
    if features.shape[1] != num_modes:
        raise ValueError(f"features mode dimension must be {num_modes}, got {features.shape[1]}.")
    if features.shape[2] != input_dim:
        raise ValueError(f"features last dimension must be {input_dim}, got {features.shape[2]}.")
    return features.shape[0], features.shape[1]


def combine_topk_expert_outputs(
    x: torch.Tensor,
    experts: nn.ModuleList,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine selected expert outputs without looping over samples."""

    output = torch.zeros_like(x)
    for expert_id, expert in enumerate(experts):
        rows, slots = torch.where(topk_indices == expert_id)
        if rows.numel() == 0:
            continue
        expert_output = expert(x.index_select(0, rows))
        weights = topk_weights[rows, slots].unsqueeze(-1)
        output = output.index_add(0, rows, expert_output * weights)
    return output


def reshape_router_tensor(tensor: torch.Tensor, batch_size: int, num_modes: int) -> torch.Tensor:
    return tensor.reshape(batch_size, num_modes, *tensor.shape[1:])


def compute_routing_statistics(
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> dict[str, Any]:
    """Compute routing statistics for validation/test analysis."""

    if router_probs.ndim != 3 or topk_indices.ndim != 3 or topk_weights.ndim != 3:
        raise ValueError("Routing tensors must have shapes [B, M, ...].")

    batch_size, num_modes, num_experts = router_probs.shape
    flat_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
    expert_counts = torch.bincount(flat_indices.reshape(-1), minlength=num_experts).to(router_probs.device)
    expert_usage = expert_counts.float() / expert_counts.sum().clamp_min(1)

    mode_counts = []
    for mode_idx in range(num_modes):
        mode_flat = topk_indices[:, mode_idx, :].reshape(-1)
        mode_counts.append(torch.bincount(mode_flat, minlength=num_experts).to(router_probs.device))
    expert_counts_by_mode = torch.stack(mode_counts, dim=0)
    expert_usage_by_mode = expert_counts_by_mode.float()
    expert_usage_by_mode = expert_usage_by_mode / expert_usage_by_mode.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1)

    entropy_by_sample_mode = -(
        router_probs * router_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    entropy = entropy_by_sample_mode.mean()
    entropy_by_mode = entropy_by_sample_mode.mean(dim=0)
    unused_experts = (expert_counts == 0).sum()
    return {
        "batch_size": batch_size,
        "expert_counts": expert_counts.detach().cpu(),
        "expert_usage": expert_usage.detach().cpu(),
        "expert_counts_by_mode": expert_counts_by_mode.detach().cpu(),
        "expert_usage_by_mode": expert_usage_by_mode.detach().cpu(),
        "router_entropy": entropy.detach().cpu(),
        "router_entropy_by_mode": entropy_by_mode.detach().cpu(),
        "unused_experts": unused_experts.detach().cpu(),
        "avg_topk_weight": topk_weights.mean().detach().cpu(),
    }
