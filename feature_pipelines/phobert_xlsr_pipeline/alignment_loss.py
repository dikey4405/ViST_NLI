from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SymmetricTextSpeechAlignmentLoss(nn.Module):
    """Symmetric InfoNCE with all matching sentences treated as positives."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        speech_embeddings: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_alignment_pair(text_embeddings, speech_embeddings)
        text = F.normalize(text_embeddings, p=2, dim=-1)
        speech = F.normalize(speech_embeddings, p=2, dim=-1)
        logits = text @ speech.transpose(0, 1) / self.temperature
        mask = _positive_mask(logits, positive_mask)
        positive_logits = logits.masked_fill(~mask, -torch.inf)
        text_loss = logits.logsumexp(dim=1) - positive_logits.logsumexp(dim=1)
        speech_loss = logits.logsumexp(dim=0) - positive_logits.logsumexp(dim=0)
        return 0.5 * (text_loss.mean() + speech_loss.mean())


def compute_alignment_metrics(
    text_embeddings: torch.Tensor,
    speech_embeddings: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return retrieval accuracy and matched/unmatched cosine statistics."""

    _validate_alignment_pair(text_embeddings, speech_embeddings)
    text = F.normalize(text_embeddings, p=2, dim=-1)
    speech = F.normalize(speech_embeddings, p=2, dim=-1)
    similarities = text @ speech.transpose(0, 1)
    mask = _positive_mask(similarities, positive_mask)
    positive_similarity = similarities.masked_select(mask).mean()
    if (~mask).any():
        negative_similarity = similarities.masked_select(~mask).mean()
    else:
        negative_similarity = similarities.sum() * 0.0
    return {
        "text_to_speech_accuracy": mask.gather(
            1, similarities.argmax(dim=1, keepdim=True)
        ).float().mean(),
        "speech_to_text_accuracy": mask.gather(
            0, similarities.argmax(dim=0, keepdim=True)
        ).float().mean(),
        "positive_cosine_similarity": positive_similarity,
        "negative_cosine_similarity": negative_similarity,
    }


def _positive_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    if mask.dtype != torch.bool or mask.shape != logits.shape or mask.device != logits.device:
        raise ValueError("positive_mask must be a boolean [B, B] tensor on the embedding device.")
    if not mask.any(dim=1).all() or not mask.any(dim=0).all():
        raise ValueError("Every text and speech query must have at least one positive.")
    return mask


def _validate_alignment_pair(
    text_embeddings: torch.Tensor,
    speech_embeddings: torch.Tensor,
) -> None:
    if text_embeddings.ndim != 2:
        raise ValueError(
            f"text_embeddings must have shape [B, D], got {tuple(text_embeddings.shape)}."
        )
    if text_embeddings.shape != speech_embeddings.shape:
        raise ValueError(
            "Text and speech embeddings must have the same shape, "
            f"got {tuple(text_embeddings.shape)} and {tuple(speech_embeddings.shape)}."
        )
