from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from ...data.schemas import INPUT_MODE_NAMES


MODE_ORDER = INPUT_MODE_NAMES
PAIR_FEATURE_COMPONENTS = ("u", "v", "abs_diff", "product")
BEST_PROJECTOR_CHECKPOINT_NAME = "best_projectors.pt"


class CacheModality(str, Enum):
    TEXT = "text"
    SPEECH = "speech"


class AudioOverlengthPolicy(str, Enum):
    TRUNCATE = "truncate"
    SKIP = "skip"
    RAISE = "raise"


@dataclass(frozen=True)
class AudioLoadResult:
    waveform: torch.Tensor
    sample_rate: int
    original_sample_rate: int
    original_num_channels: int
    was_resampled: bool
    was_truncated: bool


@dataclass(frozen=True)
class CacheManifest:
    pipeline_name: str
    split: str
    modality: str
    num_source_samples: int
    embedding_dim: int
    dtype: str
    shard_size: int
    shards: tuple[str, ...]
    processed_sample_indices: tuple[int, ...]
    skipped_sample_indices: tuple[int, ...]
    encoder_name: str
    source_data_path: str
    truncated_audio_count: int
    error_count: int
    complete: bool
    cache_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureMetadata:
    schema_version: int
    feature_source: str
    text_encoder: str
    speech_encoder: str
    text_pooling: str
    speech_pooling: str
    embedding_dim: int
    pair_feature_dim: int
    pair_feature_components: tuple[str, ...]
    mode_order: tuple[str, ...]
    projector_checkpoint: str
    alignment_temperature: float
    label_mapping: dict[str, int]
    identity_field: str
    split_counts: dict[str, int]
    skipped_counts: dict[str, int]
    split_source_files: dict[str, str]
    split_output_files: dict[str, str]
    split_order_hashes: dict[str, str]
    cache_statistics: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["projector_checkpoint"] = str(Path(self.projector_checkpoint))
        payload["pair_feature_components"] = list(self.pair_feature_components)
        payload["mode_order"] = list(self.mode_order)
        return payload
