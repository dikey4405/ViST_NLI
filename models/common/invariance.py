from __future__ import annotations

import torch
from torch import nn


class CrossModalNLIInvarianceLoss(nn.Module):
    """Average pairwise Jensen-Shannon divergence across modality predictions."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def forward(
        self,
        mode_probabilities: torch.Tensor,
        reliability: torch.Tensor | None = None,
        *,
        reliability_weighted: bool = False,
    ) -> torch.Tensor:
        statistics = self.loss_statistics(
            mode_probabilities,
            reliability,
            reliability_weighted=reliability_weighted,
        )
        return statistics["numerator"] / statistics["denominator"].clamp_min(self.eps)

    def loss_statistics(
        self,
        mode_probabilities: torch.Tensor,
        reliability: torch.Tensor | None = None,
        *,
        reliability_weighted: bool = False,
    ) -> dict[str, torch.Tensor]:
        probabilities = self._validate_and_normalize(mode_probabilities)
        batch_size, num_modes, _ = probabilities.shape
        pair_indices = torch.triu_indices(
            num_modes,
            num_modes,
            offset=1,
            device=probabilities.device,
        )
        first = probabilities[:, pair_indices[0], :]
        second = probabilities[:, pair_indices[1], :]
        midpoint = 0.5 * (first + second)
        pairwise_js = 0.5 * (
            (first * (first.log() - midpoint.log())).sum(dim=-1)
            + (second * (second.log() - midpoint.log())).sum(dim=-1)
        ).clamp_min(0.0)

        if reliability_weighted:
            self._validate_reliability(reliability, batch_size, num_modes)
            pair_weights = (
                reliability[:, pair_indices[0]] * reliability[:, pair_indices[1]]
            ).detach()
            numerator = (pairwise_js * pair_weights).sum()
            denominator = pair_weights.sum()
        else:
            numerator = pairwise_js.sum()
            denominator = pairwise_js.new_tensor(pairwise_js.numel())
        return {"numerator": numerator, "denominator": denominator}

    def _validate_and_normalize(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 3:
            raise ValueError(
                "mode_probabilities must have shape [B, M, C], "
                f"got {tuple(probabilities.shape)}."
            )
        if probabilities.shape[1] < 2 or probabilities.shape[2] < 2:
            raise ValueError("At least two modes and two classes are required.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("mode_probabilities must contain only finite values.")
        if torch.any(probabilities < 0):
            raise ValueError("mode_probabilities must be non-negative.")

        normalized = probabilities.clamp_min(self.eps)
        return normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @staticmethod
    def _validate_reliability(
        reliability: torch.Tensor | None,
        batch_size: int,
        num_modes: int,
    ) -> None:
        if reliability is None:
            raise ValueError("reliability is required for reliability-weighted invariance.")
        if reliability.shape != (batch_size, num_modes):
            raise ValueError(
                f"reliability must have shape {(batch_size, num_modes)}, "
                f"got {tuple(reliability.shape)}."
            )
        if not torch.isfinite(reliability).all():
            raise ValueError("reliability must contain only finite values.")
        if torch.any((reliability < 0) | (reliability > 1)):
            raise ValueError("reliability values must be in [0, 1].")
