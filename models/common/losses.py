from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_load_balancing_loss(
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_routed_experts: int,
) -> torch.Tensor:
    """Differentiable load-balancing loss over routed experts only."""

    if router_probs.shape[-1] != num_routed_experts:
        raise ValueError(
            f"router_probs last dim must be {num_routed_experts}, got {tuple(router_probs.shape)}."
        )

    flat_probs = router_probs.reshape(-1, num_routed_experts)
    flat_indices = topk_indices.reshape(-1, topk_indices.shape[-1])

    token_fraction = flat_probs.mean(dim=0)
    selected = F.one_hot(flat_indices, num_classes=num_routed_experts).float().sum(dim=1)
    selected_fraction = selected.mean(dim=0) / flat_indices.shape[-1]
    return num_routed_experts * torch.sum(token_fraction * selected_fraction)


def compute_mode_auxiliary_loss(
    mode_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Average supervised NLI cross entropy over all modes of each sample."""

    if mode_logits.ndim != 3:
        raise ValueError(
            f"mode_logits must have shape [B, M, C], got {tuple(mode_logits.shape)}."
        )
    if labels.ndim != 1 or labels.shape[0] != mode_logits.shape[0]:
        raise ValueError(
            f"labels must have shape [{mode_logits.shape[0]}], got {tuple(labels.shape)}."
        )
    expanded_labels = labels.unsqueeze(1).expand(-1, mode_logits.shape[1])
    return F.cross_entropy(
        mode_logits.reshape(-1, mode_logits.shape[-1]),
        expanded_labels.reshape(-1),
    )
