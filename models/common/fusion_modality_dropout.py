from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


EXPECTED_NUM_MODES = 4
TT_INDEX = 0
ST_INDEX = 2


class FusionModalityDropout:
    """Mask selected modes after attention scoring during training only."""

    def __init__(
        self,
        *,
        drop_tt_prob: float,
        drop_tt_st_prob: float,
    ) -> None:
        self.drop_tt_prob = _validate_probability(
            "drop_tt_prob",
            drop_tt_prob,
        )
        self.drop_tt_st_prob = _validate_probability(
            "drop_tt_st_prob",
            drop_tt_st_prob,
        )
        probability_sum = self.drop_tt_prob + self.drop_tt_st_prob
        if probability_sum > 1.0:
            raise ValueError(
                "drop_tt_prob + drop_tt_st_prob must not exceed 1, "
                f"got {probability_sum}."
            )

    @property
    def is_noop(self) -> bool:
        return self.drop_tt_prob == 0.0 and self.drop_tt_st_prob == 0.0

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not module.training or self.is_noop:
            return None

        mode_embeddings, attention_weights = self._validate_hook_tensors(
            inputs,
            output,
        )
        keep_mask = self._sample_keep_mask(
            batch_size=mode_embeddings.shape[0],
            device=mode_embeddings.device,
        )
        normalized_weights = self._mask_and_normalize(
            attention_weights,
            keep_mask,
        )
        fused = torch.sum(
            normalized_weights.unsqueeze(-1).to(mode_embeddings.dtype)
            * mode_embeddings,
            dim=1,
        )
        if not torch.isfinite(fused).all():
            raise FloatingPointError(
                "Fusion modality dropout produced non-finite fused embeddings."
            )
        return fused, normalized_weights

    def _sample_keep_mask(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        samples = torch.rand(batch_size, device=device)
        drop_tt = samples < self.drop_tt_prob
        drop_tt_st = (
            (samples >= self.drop_tt_prob)
            & (samples < self.drop_tt_prob + self.drop_tt_st_prob)
        )

        keep_mask = torch.ones(
            batch_size,
            EXPECTED_NUM_MODES,
            dtype=torch.bool,
            device=device,
        )
        keep_mask[drop_tt, TT_INDEX] = False
        keep_mask[drop_tt_st, TT_INDEX] = False
        keep_mask[drop_tt_st, ST_INDEX] = False
        if not keep_mask.any(dim=1).all():
            raise RuntimeError("Fusion modality dropout masked every mode in a sample.")
        return keep_mask

    @staticmethod
    def _mask_and_normalize(
        attention_weights: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.isfinite(attention_weights).all():
            raise FloatingPointError("Attention weights contain NaN or Inf values.")

        normalization_dtype = (
            torch.float32
            if attention_weights.dtype in {torch.float16, torch.bfloat16}
            else attention_weights.dtype
        )
        working_weights = attention_weights.to(normalization_dtype)
        masked_weights = torch.where(
            keep_mask,
            working_weights,
            torch.zeros_like(working_weights),
        )
        weight_sums = masked_weights.sum(dim=1, keepdim=True)

        kept_counts = keep_mask.sum(dim=1, keepdim=True).to(normalization_dtype)
        uniform_fallback = keep_mask.to(normalization_dtype) / kept_counts
        minimum = torch.finfo(normalization_dtype).tiny
        normalized_weights = torch.where(
            weight_sums > 0,
            masked_weights / weight_sums.clamp_min(minimum),
            uniform_fallback,
        )
        normalized_weights = normalized_weights.to(attention_weights.dtype)
        normalized_weights = torch.where(
            keep_mask,
            normalized_weights,
            torch.zeros_like(normalized_weights),
        )
        if not torch.isfinite(normalized_weights).all():
            raise FloatingPointError(
                "Fusion modality dropout produced non-finite attention weights."
            )
        return normalized_weights

    @staticmethod
    def _validate_hook_tensors(
        inputs: tuple[Any, ...],
        output: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError(
                "ModeAttentionPooling hook requires mode embeddings as its first input."
            )
        mode_embeddings = inputs[0]
        if mode_embeddings.ndim != 3:
            raise ValueError(
                "mode embeddings must have shape [B, 4, H], "
                f"got {tuple(mode_embeddings.shape)}."
            )
        if mode_embeddings.shape[1] != EXPECTED_NUM_MODES:
            raise ValueError(
                f"Fusion modality dropout requires exactly {EXPECTED_NUM_MODES} modes, "
                f"got {mode_embeddings.shape[1]}."
            )
        if (
            not isinstance(output, Sequence)
            or len(output) != 2
            or not isinstance(output[1], torch.Tensor)
        ):
            raise TypeError(
                "ModeAttentionPooling hook requires output (pooled, attention_weights)."
            )
        attention_weights = output[1]
        expected_shape = mode_embeddings.shape[:2]
        if attention_weights.shape != expected_shape:
            raise ValueError(
                f"attention weights must have shape {tuple(expected_shape)}, "
                f"got {tuple(attention_weights.shape)}."
            )
        if attention_weights.device != mode_embeddings.device:
            raise ValueError(
                "attention weights and mode embeddings must be on the same device."
            )
        return mode_embeddings, attention_weights


def register_fusion_modality_dropout(
    mode_pooling: nn.Module,
    *,
    num_modes: int,
    drop_tt_prob: float = 0.30,
    drop_tt_st_prob: float = 0.10,
) -> RemovableHandle | None:
    """Register training-only modality dropout on an attention pooling module."""

    if num_modes != EXPECTED_NUM_MODES:
        raise ValueError(
            f"Fusion modality dropout requires num_modes={EXPECTED_NUM_MODES}, "
            f"got {num_modes}."
        )
    hook = FusionModalityDropout(
        drop_tt_prob=drop_tt_prob,
        drop_tt_st_prob=drop_tt_st_prob,
    )
    if hook.is_noop:
        return None
    return mode_pooling.register_forward_hook(hook)


def _validate_probability(name: str, value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}.")
    return probability
