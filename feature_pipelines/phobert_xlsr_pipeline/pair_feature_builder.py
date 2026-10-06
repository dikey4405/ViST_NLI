from __future__ import annotations

import torch

from ...data.schemas import INPUT_MODE_NAMES, InputMode
from ...pair_features import build_pair_feature


def build_four_mode_features(
    premise_text: torch.Tensor,
    premise_speech: torch.Tensor,
    hypothesis_text: torch.Tensor,
    hypothesis_speech: torch.Tensor,
    *,
    embedding_dim: int = 1024,
) -> torch.Tensor:
    """Build ordered [B, 4, 4*D] NLI pair features without cross-mode fusion."""

    embeddings = (premise_text, premise_speech, hypothesis_text, hypothesis_speech)
    reference_shape = premise_text.shape
    for name, embedding in zip(
        ("premise_text", "premise_speech", "hypothesis_text", "hypothesis_speech"),
        embeddings,
    ):
        if embedding.ndim != 2 or embedding.shape[-1] != embedding_dim:
            raise ValueError(
                f"{name} must have shape [B, {embedding_dim}], got {tuple(embedding.shape)}."
            )
        if embedding.shape != reference_shape:
            raise ValueError("All four projected embedding batches must have the same shape.")

    mode_pairs = {
        InputMode.TEXT_TEXT.value: (premise_text, hypothesis_text),
        InputMode.TEXT_SPEECH.value: (premise_text, hypothesis_speech),
        InputMode.SPEECH_TEXT.value: (premise_speech, hypothesis_text),
        InputMode.SPEECH_SPEECH.value: (premise_speech, hypothesis_speech),
    }
    features = torch.stack(
        [
            build_pair_feature(
                *mode_pairs[mode],
                expected_embedding_dim=embedding_dim,
            )
            for mode in INPUT_MODE_NAMES
        ],
        dim=1,
    )
    if features.shape[1] != len(INPUT_MODE_NAMES):
        raise RuntimeError(f"Expected {len(INPUT_MODE_NAMES)} modes, got {features.shape[1]}.")
    return features
