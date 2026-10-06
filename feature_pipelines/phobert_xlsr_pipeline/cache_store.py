from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from ...data.utils import read_json_or_jsonl
from .cache_identity import compute_cache_fingerprint, sentence_key
from .config import PhoBERTXLSRConfig
from .schemas import CacheManifest, CacheModality
from .utils import atomic_json_save, resolve_torch_dtype, torch_load


EMBEDDING_DIM = 1024


def modality_cache_directory(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> Path:
    return config.cache.directory / split / modality.value


def manifest_path(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> Path:
    return modality_cache_directory(config, split, modality) / "manifest.json"


def load_manifest(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> CacheManifest | None:
    path = manifest_path(config, split, modality)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache manifest must contain a JSON object: {path}")
    allowed = {field.name for field in fields(CacheManifest)}
    values = {key: value for key, value in payload.items() if key in allowed}
    required = allowed - {"source_data_path", "truncated_audio_count", "error_count"}
    missing = required - values.keys()
    if missing:
        raise ValueError(
            f"Cache manifest is missing fields {sorted(missing)}: {path}. "
            "Set cache.overwrite=true to rebuild this legacy or incomplete cache."
        )
    values["shards"] = tuple(values["shards"])
    values["processed_sample_indices"] = tuple(values["processed_sample_indices"])
    values["skipped_sample_indices"] = tuple(values["skipped_sample_indices"])
    values.setdefault("source_data_path", str(config.data.split_path(split)))
    values.setdefault("truncated_audio_count", 0)
    values.setdefault("error_count", len(values["skipped_sample_indices"]))
    return CacheManifest(**values)


def save_manifest(
    manifest: CacheManifest,
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> None:
    atomic_json_save(manifest.to_dict(), manifest_path(config, split, modality))


def create_cache_manifest(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
    encoder_name: str,
    num_source_samples: int,
    *,
    cache_fingerprint: str | None = None,
) -> CacheManifest:
    return build_cache_manifest(
        config,
        split,
        modality,
        encoder_name,
        num_source_samples=num_source_samples,
        shards=[],
        processed=set(),
        skipped=set(),
        truncated_audio_count=0,
        error_count=0,
        complete=False,
        cache_fingerprint=cache_fingerprint,
    )


def build_cache_manifest(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
    encoder_name: str,
    *,
    num_source_samples: int,
    shards: list[str],
    processed: set[int],
    skipped: set[int],
    truncated_audio_count: int,
    error_count: int,
    complete: bool,
    cache_fingerprint: str | None = None,
) -> CacheManifest:
    return CacheManifest(
        pipeline_name=config.pipeline_name,
        split=split,
        modality=modality.value,
        num_source_samples=num_source_samples,
        embedding_dim=config.projector.input_dim,
        dtype=config.cache.dtype,
        shard_size=config.cache.shard_size,
        shards=tuple(shards),
        processed_sample_indices=tuple(sorted(processed)),
        skipped_sample_indices=tuple(sorted(skipped)),
        encoder_name=encoder_name,
        source_data_path=str(config.data.split_path(split).resolve()),
        truncated_audio_count=truncated_audio_count,
        error_count=error_count,
        complete=complete,
        cache_fingerprint=(
            cache_fingerprint or compute_cache_fingerprint(config, split, modality)
        ),
    )


def validate_resume_manifest(
    manifest: CacheManifest,
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
    encoder_name: str,
    *,
    cache_fingerprint: str | None = None,
) -> None:
    expected = {
        "pipeline_name": config.pipeline_name,
        "split": split,
        "modality": modality.value,
        "embedding_dim": config.projector.input_dim,
        "dtype": config.cache.dtype,
        "shard_size": config.cache.shard_size,
        "encoder_name": encoder_name,
        "source_data_path": str(config.data.split_path(split).resolve()),
        "cache_fingerprint": (
            cache_fingerprint or compute_cache_fingerprint(config, split, modality)
        ),
    }
    mismatches = {
        key: (getattr(manifest, key), value)
        for key, value in expected.items()
        if getattr(manifest, key) != value
    }
    if mismatches:
        raise ValueError(
            f"Cannot resume incompatible cache for split='{split}', "
            f"modality='{modality.value}': {mismatches}. "
            "Set cache.overwrite=true to rebuild it."
        )


def validate_manifest_shards(cache_directory: Path, manifest: CacheManifest) -> None:
    listed = set(manifest.shards)
    actual = {path.name for path in cache_directory.glob("shard_*.pt")}
    if listed != actual:
        raise RuntimeError(
            f"Cache manifest/shard mismatch in {cache_directory}: "
            f"missing={sorted(listed - actual)}, unlisted={sorted(actual - listed)}."
        )


def load_merged_cached_embeddings(
    config: PhoBERTXLSRConfig,
    split: str,
) -> dict[str, Any]:
    """Load complete text/speech caches and align rows by sample_index."""

    text_manifest = _require_manifest(config, split, CacheModality.TEXT)
    speech_manifest = _require_manifest(config, split, CacheModality.SPEECH)
    if text_manifest.num_source_samples != speech_manifest.num_source_samples:
        raise ValueError(f"Text and speech cache source counts differ for split '{split}'.")
    text_cache = _load_modality_cache(config, split, CacheModality.TEXT, text_manifest)
    speech_cache = _load_modality_cache(
        config,
        split,
        CacheModality.SPEECH,
        speech_manifest,
    )
    text_positions = {
        int(sample_index): position
        for position, sample_index in enumerate(text_cache["sample_indices"].tolist())
    }
    speech_positions = {
        int(sample_index): position
        for position, sample_index in enumerate(speech_cache["sample_indices"].tolist())
    }
    common_indices = sorted(set(text_positions) & set(speech_positions))
    if not common_indices:
        raise ValueError(f"No common text/speech cached samples for split '{split}'.")

    text_rows = torch.tensor(
        [text_positions[index] for index in common_indices], dtype=torch.long
    )
    speech_rows = torch.tensor(
        [speech_positions[index] for index in common_indices], dtype=torch.long
    )
    ids = [text_cache["ids"][position] for position in text_rows.tolist()]
    speech_ids = [speech_cache["ids"][position] for position in speech_rows.tolist()]
    labels = [text_cache["labels"][position] for position in text_rows.tolist()]
    speech_labels = [speech_cache["labels"][position] for position in speech_rows.tolist()]
    if ids != speech_ids or labels != speech_labels:
        raise ValueError(f"Text and speech caches are misaligned for split '{split}'.")

    records = read_json_or_jsonl(config.data.split_path(split))
    if common_indices[-1] >= len(records):
        raise ValueError(f"Cache indices exceed the source sample count for split '{split}'.")
    sentence_keys = {
        f"{side}_sentence_keys": [sentence_key(records[index][side]) for index in common_indices]
        for side in ("premise", "hypothesis")
    }

    return {
        **sentence_keys,
        "ids": ids,
        "labels": labels,
        "sample_indices": torch.tensor(common_indices, dtype=torch.long),
        "premise_text_embeddings": text_cache["premise_embeddings"].index_select(
            0, text_rows
        ),
        "hypothesis_text_embeddings": text_cache["hypothesis_embeddings"].index_select(
            0, text_rows
        ),
        "premise_speech_embeddings": speech_cache["premise_embeddings"].index_select(
            0, speech_rows
        ),
        "hypothesis_speech_embeddings": speech_cache["hypothesis_embeddings"].index_select(
            0, speech_rows
        ),
        "skipped_count": text_manifest.num_source_samples - len(common_indices),
    }


def _require_manifest(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> CacheManifest:
    manifest = load_manifest(config, split, modality)
    if manifest is None or not manifest.complete:
        raise FileNotFoundError(
            f"Complete {modality.value} cache not found for split '{split}'."
        )
    expected_encoder = (
        config.text_encoder.model_name
        if modality == CacheModality.TEXT
        else config.speech_encoder.model_name
    )
    expected = {
        "pipeline_name": config.pipeline_name,
        "split": split,
        "modality": modality.value,
        "embedding_dim": config.projector.input_dim,
        "dtype": config.cache.dtype,
        "encoder_name": expected_encoder,
        "source_data_path": str(config.data.split_path(split).resolve()),
        "cache_fingerprint": compute_cache_fingerprint(config, split, modality),
    }
    mismatches = {
        key: (getattr(manifest, key), value)
        for key, value in expected.items()
        if getattr(manifest, key) != value
    }
    if mismatches:
        raise ValueError(
            f"Cache manifest is incompatible for split='{split}', "
            f"modality='{modality.value}': {mismatches}."
        )
    return manifest


def _load_modality_cache(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
    manifest: CacheManifest,
) -> dict[str, Any]:
    payloads = []
    for shard in manifest.shards:
        shard_path = modality_cache_directory(config, split, modality) / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Cache shard listed in manifest is missing: {shard_path}")
        payload = torch_load(shard_path)
        _validate_shard_payload(payload, shard_path)
        payloads.append(payload)
    if not payloads:
        raise ValueError(f"Cache has no shards: split='{split}', modality='{modality.value}'.")

    merged = {
        "ids": [item for payload in payloads for item in payload["ids"]],
        "labels": [item for payload in payloads for item in payload["labels"]],
        "sample_indices": torch.cat([payload["sample_indices"] for payload in payloads]),
        "premise_embeddings": torch.cat(
            [payload["premise_embeddings"] for payload in payloads]
        ),
        "hypothesis_embeddings": torch.cat(
            [payload["hypothesis_embeddings"] for payload in payloads]
        ),
    }
    _validate_merged_cache(merged, manifest, split=split, modality=modality)
    return merged


def _validate_shard_payload(payload: Any, path: Path) -> None:
    required_keys = {
        "ids",
        "labels",
        "sample_indices",
        "premise_embeddings",
        "hypothesis_embeddings",
    }
    if not isinstance(payload, dict):
        raise ValueError(f"Cache shard must contain a dictionary: {path}")
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"Cache shard is missing keys {sorted(missing)}: {path}")


def _validate_merged_cache(
    cache: dict[str, Any],
    manifest: CacheManifest,
    *,
    split: str,
    modality: CacheModality,
) -> None:
    sample_indices = cache["sample_indices"]
    premise = cache["premise_embeddings"]
    hypothesis = cache["hypothesis_embeddings"]
    num_samples = len(cache["ids"])
    context = f"split='{split}', modality='{modality.value}'"
    if len(cache["labels"]) != num_samples or sample_indices.shape != (num_samples,):
        raise ValueError(f"Cache metadata lengths are inconsistent for {context}.")
    expected_shape = (num_samples, EMBEDDING_DIM)
    if premise.shape != expected_shape or hypothesis.shape != expected_shape:
        raise ValueError(
            f"Cached embeddings must have shape {expected_shape} for {context}; "
            f"got {tuple(premise.shape)} and {tuple(hypothesis.shape)}."
        )
    expected_dtype = resolve_torch_dtype(manifest.dtype)
    if premise.dtype != expected_dtype or hypothesis.dtype != expected_dtype:
        raise ValueError(
            f"Cached embeddings must use dtype {expected_dtype} for {context}; "
            f"got {premise.dtype} and {hypothesis.dtype}."
        )
    if not torch.isfinite(premise).all() or not torch.isfinite(hypothesis).all():
        raise ValueError(f"Cached embeddings contain NaN or Inf for {context}.")
    indices = [int(value) for value in sample_indices.tolist()]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"Cached sample_index values must be unique and sorted for {context}.")
    if tuple(indices) != manifest.processed_sample_indices:
        raise ValueError(f"Cache rows do not match processed_sample_indices for {context}.")
