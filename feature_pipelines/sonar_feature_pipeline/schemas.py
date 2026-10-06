from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...data.schemas import InputMode, ModalityType, NLIInputPair, NLISample, RawInputValue


@dataclass(frozen=True)
class NLIFeatureSample:
    """Feature vectors for all input modes of one NLI sample."""

    id: str | None
    label: str
    features: dict[str, torch.Tensor]
    sample_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "features": self.features,
        }
        if self.sample_index is not None:
            payload["sample_index"] = self.sample_index
        return payload


__all__ = [
    "InputMode",
    "ModalityType",
    "NLIInputPair",
    "NLIFeatureSample",
    "NLISample",
    "RawInputValue",
]
