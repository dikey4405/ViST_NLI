from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..data.config import DEFAULT_LABEL_MAPPING
from ..data.schemas import INPUT_MODE_NAMES


class FeatureTensorDataset(Dataset):
    """Dataset over one saved four-mode feature source (SONAR or PhoBERT/XLS-R)."""

    def __init__(
        self,
        feature_path: str | Path,
        *,
        label_mapping: dict[str, int] | None = None,
        expected_feature_dim: int = 4096,
    ) -> None:
        self.feature_path = Path(feature_path)
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_path}")
        self.label_mapping = label_mapping or dict(DEFAULT_LABEL_MAPPING)
        if expected_feature_dim < 1:
            raise ValueError(
                f"expected_feature_dim must be positive, got {expected_feature_dim}."
            )
        self.expected_feature_dim = expected_feature_dim
        payload = torch.load(self.feature_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list):
            raise ValueError(f"Feature file must contain a list of samples: {self.feature_path}")
        if not payload:
            raise ValueError(f"Feature file contains no samples: {self.feature_path}")
        for index, sample in enumerate(payload):
            self._validate_sample(sample, index)
        self.samples = payload

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        features_by_mode = sample["features"]
        features = torch.stack([features_by_mode[mode].float() for mode in INPUT_MODE_NAMES], dim=0)
        label = sample["label"]
        label_id = self.label_mapping[label] if isinstance(label, str) else int(label)
        return {
            "id": sample.get("id"),
            "sample_index": sample.get("sample_index", index),
            "features": features,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_text": label,
        }

    def _validate_sample(self, sample: Any, index: int) -> None:
        context = f"sample index {index} in {self.feature_path}"
        if not isinstance(sample, dict):
            raise ValueError(f"Feature sample must be a dictionary at {context}.")
        features_by_mode = sample.get("features")
        if not isinstance(features_by_mode, dict):
            raise ValueError(f"Missing feature dictionary at {context}.")
        if set(features_by_mode) != set(INPUT_MODE_NAMES):
            raise ValueError(
                f"Feature modes must be exactly {INPUT_MODE_NAMES} at {context}, "
                f"got {tuple(features_by_mode)}."
            )
        for mode in INPUT_MODE_NAMES:
            feature = features_by_mode[mode]
            expected_shape = (self.expected_feature_dim,)
            if not isinstance(feature, torch.Tensor) or feature.shape != expected_shape:
                shape = tuple(feature.shape) if isinstance(feature, torch.Tensor) else None
                raise ValueError(
                    f"Feature '{mode}' must have shape {expected_shape} at {context}, "
                    f"got {shape}."
                )
            if not torch.isfinite(feature).all():
                raise ValueError(f"Feature '{mode}' contains NaN or Inf at {context}.")
        label = sample.get("label")
        if isinstance(label, str) and label not in self.label_mapping:
            raise ValueError(f"Unknown label '{label}' at {context}.")
        if isinstance(label, int) and label not in self.label_mapping.values():
            raise ValueError(f"Unknown integer label '{label}' at {context}.")
        if not isinstance(label, (str, int)) or isinstance(label, bool):
            raise ValueError(f"Label must be a string or integer at {context}, got {label!r}.")


class FeatureBatchCollator:
    """Collate saved feature samples into configurable [B, 4, F] tensors."""

    def __init__(self, expected_feature_dim: int = 4096) -> None:
        if expected_feature_dim < 1:
            raise ValueError(
                f"expected_feature_dim must be positive, got {expected_feature_dim}."
            )
        self.expected_feature_dim = expected_feature_dim

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        features = torch.stack([sample["features"] for sample in samples], dim=0)
        expected_shape = (
            len(samples),
            len(INPUT_MODE_NAMES),
            self.expected_feature_dim,
        )
        if features.shape != expected_shape:
            raise ValueError(
                f"Feature batch must have shape {expected_shape}, got {tuple(features.shape)}."
            )
        return {
            "ids": [sample["id"] for sample in samples],
            "sample_indices": [sample["sample_index"] for sample in samples],
            "features": features,
            "labels": torch.stack([sample["label"] for sample in samples], dim=0),
            "label_texts": [sample["label_text"] for sample in samples],
        }
