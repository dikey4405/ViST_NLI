from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from .pooling import masked_mean_pooling
from .utils import autocast_context, freeze_module, resolve_device


class PhoBERTTextEncoder(nn.Module):
    """Frozen PhoBERT encoder with padding-aware masked mean pooling."""

    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        expected_output_dim: int = 1024,
        freeze: bool = True,
        device: str = "auto",
        mixed_precision: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for PhoBERT feature extraction.") from exc

        self.device = resolve_device(device)
        self.max_length = max_length
        self.expected_output_dim = expected_output_dim
        self.freeze = freeze
        self.mixed_precision = mixed_precision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        model_limit = int(getattr(self.model.config, "max_position_embeddings", max_length))
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", model_limit))
        if tokenizer_limit > 1_000_000:
            tokenizer_limit = model_limit
        supported_limit = min(model_limit, tokenizer_limit)
        if max_length > supported_limit:
            raise ValueError(
                f"Configured max_length={max_length} exceeds model/tokenizer limit "
                f"{supported_limit}."
            )
        if freeze:
            freeze_module(self.model)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must contain at least one item.")
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        gradient_context = torch.inference_mode() if self.freeze else nullcontext()
        mixed_precision_context = autocast_context(self.device, self.mixed_precision)
        with gradient_context, mixed_precision_context:
            outputs: Any = self.model(**tokens)
            hidden_states = outputs.last_hidden_state
            pooled = masked_mean_pooling(hidden_states, tokens["attention_mask"])
        if pooled.ndim != 2 or pooled.shape[-1] != self.expected_output_dim:
            raise ValueError(
                f"PhoBERT pooled output must have shape [B, {self.expected_output_dim}], "
                f"got {tuple(pooled.shape)}."
            )
        return pooled.float()
