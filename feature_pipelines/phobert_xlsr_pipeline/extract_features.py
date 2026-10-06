from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ...data.config import DEFAULT_LABEL_MAPPING
from .cache_store import load_manifest, load_merged_cached_embeddings
from .config import PhoBERTXLSRConfig, load_pipeline_config
from .pair_feature_builder import build_four_mode_features
from .projectors import SpeechProjector, TextProjector, build_projector_pair
from .schemas import (
    BEST_PROJECTOR_CHECKPOINT_NAME,
    CacheModality,
    FeatureMetadata,
    MODE_ORDER,
    PAIR_FEATURE_COMPONENTS,
)
from .utils import (
    atomic_json_save,
    atomic_torch_save,
    compute_sample_order_hash,
    ensure_directory,
    get_logger,
    peak_gpu_memory_mb,
    resolve_device,
    resolve_torch_dtype,
    set_random_seed,
    torch_load,
)


LOGGER = get_logger(__name__)


def extract_split_features(
    config: PhoBERTXLSRConfig,
    split: str,
    *,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Project cached embeddings and save four independent pair features per sample."""

    set_random_seed(config.seed)
    device = resolve_device(config.device)
    output_dtype = resolve_torch_dtype(config.output.dtype)
    output_directory = ensure_directory(config.output.directory)
    output_path = output_directory / f"{split}.pt"
    if output_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Feature output already exists: {output_path}. "
            "Set output.overwrite=true to replace it."
        )

    resolved_checkpoint = Path(
        checkpoint_path or config.checkpoint.directory / BEST_PROJECTOR_CHECKPOINT_NAME
    )
    text_projector, speech_projector = load_projectors(
        config,
        resolved_checkpoint,
        device=device,
    )
    cache = load_merged_cached_embeddings(config, split)
    num_samples = len(cache["ids"])
    if num_samples == 0:
        raise ValueError(f"No common cached samples are available for split '{split}'.")

    LOGGER.info(
        "extract_start split=%s samples=%d checkpoint=%s device=%s output_dtype=%s",
        split,
        num_samples,
        resolved_checkpoint,
        device,
        config.output.dtype,
    )
    results: list[dict[str, Any]] = []
    batch_size = config.alignment.batch_size
    started_at = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(range(0, num_samples, batch_size), desc=f"extract {split}", unit="batch")
        for start in progress:
            stop = min(start + batch_size, num_samples)
            premise_text = _cache_slice(cache, "premise_text_embeddings", start, stop, device)
            premise_speech = _cache_slice(
                cache, "premise_speech_embeddings", start, stop, device
            )
            hypothesis_text = _cache_slice(
                cache, "hypothesis_text_embeddings", start, stop, device
            )
            hypothesis_speech = _cache_slice(
                cache, "hypothesis_speech_embeddings", start, stop, device
            )

            projected_premise_text = text_projector(premise_text)
            projected_premise_speech = speech_projector(premise_speech)
            projected_hypothesis_text = text_projector(hypothesis_text)
            projected_hypothesis_speech = speech_projector(hypothesis_speech)
            features = build_four_mode_features(
                projected_premise_text,
                projected_premise_speech,
                projected_hypothesis_text,
                projected_hypothesis_speech,
                embedding_dim=config.projector.output_dim,
            )
            expected_shape = (stop - start, len(MODE_ORDER), config.pair_feature.output_dim)
            if features.shape != expected_shape:
                raise RuntimeError(
                    f"Four-mode feature batch must have shape {expected_shape}, "
                    f"got {tuple(features.shape)}."
                )
            if not torch.isfinite(features).all():
                raise ValueError(
                    f"Non-finite feature detected in split '{split}', rows {start}:{stop}."
                )
            features = features.detach().cpu().to(output_dtype)
            for row_index in range(stop - start):
                results.append(
                    {
                        "id": cache["ids"][start + row_index],
                        "label": cache["labels"][start + row_index],
                        "sample_index": int(cache["sample_indices"][start + row_index]),
                        "features": {
                            mode: features[row_index, mode_index]
                            for mode_index, mode in enumerate(MODE_ORDER)
                        },
                    }
                )

    if len(results) != num_samples:
        raise RuntimeError(f"Expected {num_samples} output samples, built {len(results)}.")
    atomic_torch_save(results, output_path)
    _update_metadata(
        config,
        split,
        cache,
        output_path,
        resolved_checkpoint,
    )
    LOGGER.info(
        "extract_done split=%s logical_shape=%s output=%s elapsed_seconds=%.2f "
        "peak_gpu_memory_mb=%s",
        split,
        (num_samples, len(MODE_ORDER), config.pair_feature.output_dim),
        output_path,
        time.perf_counter() - started_at,
        peak_gpu_memory_mb(device),
    )
    return output_path


def load_projectors(
    config: PhoBERTXLSRConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[TextProjector, SpeechProjector]:
    """Load the two trained projectors without loading either base encoder."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path)
    required_keys = {
        "text_projector_state_dict",
        "speech_projector_state_dict",
        "config",
        "best_validation_loss",
        "epoch",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(f"Projector checkpoint is missing keys: {sorted(missing_keys)}.")
    _validate_checkpoint_config(checkpoint["config"], config)

    text_projector, speech_projector = build_projector_pair(
        input_dim=config.projector.input_dim,
        hidden_dim=config.projector.hidden_dim,
        output_dim=config.projector.output_dim,
        dropout=config.projector.dropout,
        device=device,
    )
    text_projector.load_state_dict(checkpoint["text_projector_state_dict"], strict=True)
    speech_projector.load_state_dict(checkpoint["speech_projector_state_dict"], strict=True)
    text_projector.eval()
    speech_projector.eval()
    return text_projector, speech_projector


def _cache_slice(
    cache: dict[str, Any],
    key: str,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    return cache[key][start:stop].to(device=device, dtype=torch.float32, non_blocking=True)


def _validate_checkpoint_config(
    checkpoint_config: dict[str, Any],
    config: PhoBERTXLSRConfig,
) -> None:
    expected = {
        "pipeline_name": config.pipeline_name,
        "text_encoder.model_name": config.text_encoder.model_name,
        "speech_encoder.model_name": config.speech_encoder.model_name,
        "projector.input_dim": config.projector.input_dim,
        "projector.hidden_dim": config.projector.hidden_dim,
        "projector.output_dim": config.projector.output_dim,
    }
    actual = {
        "pipeline_name": checkpoint_config.get("pipeline_name"),
        "text_encoder.model_name": checkpoint_config.get("text_encoder", {}).get("model_name"),
        "speech_encoder.model_name": checkpoint_config.get("speech_encoder", {}).get(
            "model_name"
        ),
        "projector.input_dim": checkpoint_config.get("projector", {}).get("input_dim"),
        "projector.hidden_dim": checkpoint_config.get("projector", {}).get("hidden_dim"),
        "projector.output_dim": checkpoint_config.get("projector", {}).get("output_dim"),
    }
    mismatches = {
        key: (actual[key], expected_value)
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Projector checkpoint is incompatible with the active config: {mismatches}."
        )


def _update_metadata(
    config: PhoBERTXLSRConfig,
    split: str,
    cache: dict[str, Any],
    output_path: Path,
    checkpoint_path: Path,
) -> None:
    metadata_path = config.output.directory / "metadata.json"
    existing = _load_existing_metadata(metadata_path)
    split_counts = dict(existing.get("split_counts", {}))
    skipped_counts = dict(existing.get("skipped_counts", {}))
    split_source_files = dict(existing.get("split_source_files", {}))
    split_output_files = dict(existing.get("split_output_files", {}))
    split_order_hashes = dict(existing.get("split_order_hashes", {}))
    cache_statistics = dict(existing.get("cache_statistics", {}))

    split_counts[split] = len(cache["ids"])
    skipped_counts[split] = int(cache["skipped_count"])
    split_source_files[split] = str(config.data.split_path(split).resolve())
    split_output_files[split] = str(output_path.resolve())
    split_order_hashes[split] = compute_sample_order_hash(
        cache["sample_indices"], cache["ids"], cache["labels"]
    )
    text_manifest = load_manifest(config, split, CacheModality.TEXT)
    speech_manifest = load_manifest(config, split, CacheModality.SPEECH)
    if text_manifest is None or speech_manifest is None:
        raise FileNotFoundError(f"Cache manifests disappeared while extracting split '{split}'.")
    cache_statistics[split] = {
        "source_samples": text_manifest.num_source_samples,
        "saved_samples": len(cache["ids"]),
        "skipped_samples": int(cache["skipped_count"]),
        "text_errors": text_manifest.error_count,
        "speech_errors": speech_manifest.error_count,
        "truncated_audio": speech_manifest.truncated_audio_count,
    }

    metadata = FeatureMetadata(
        schema_version=1,
        feature_source=config.pipeline_name,
        text_encoder=config.text_encoder.model_name,
        speech_encoder=config.speech_encoder.model_name,
        text_pooling=config.text_encoder.pooling,
        speech_pooling=config.speech_encoder.pooling,
        embedding_dim=config.projector.output_dim,
        pair_feature_dim=config.pair_feature.output_dim,
        pair_feature_components=PAIR_FEATURE_COMPONENTS,
        mode_order=MODE_ORDER,
        projector_checkpoint=str(checkpoint_path.resolve()),
        alignment_temperature=config.alignment.temperature,
        label_mapping=dict(DEFAULT_LABEL_MAPPING),
        identity_field="sample_index",
        split_counts=split_counts,
        skipped_counts=skipped_counts,
        split_source_files=split_source_files,
        split_output_files=split_output_files,
        split_order_hashes=split_order_hashes,
        cache_statistics=cache_statistics,
    )
    _validate_existing_metadata(existing, metadata.to_dict())
    atomic_json_save(metadata.to_dict(), metadata_path)


def _load_existing_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Feature metadata must be a JSON object: {path}")
    return payload


def _validate_existing_metadata(existing: dict[str, Any], current: dict[str, Any]) -> None:
    immutable_keys = (
        "schema_version",
        "feature_source",
        "text_encoder",
        "speech_encoder",
        "text_pooling",
        "speech_pooling",
        "embedding_dim",
        "pair_feature_dim",
        "pair_feature_components",
        "mode_order",
        "projector_checkpoint",
        "alignment_temperature",
        "label_mapping",
        "identity_field",
    )
    mismatches = {
        key: (existing[key], current[key])
        for key in immutable_keys
        if key in existing and existing[key] != current[key]
    }
    if mismatches:
        raise ValueError(
            "Existing feature metadata belongs to an incompatible extraction run: "
            f"{mismatches}. Use a separate output directory or enable a clean overwrite."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned PhoBERT/XLS-R NLI features.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    extract_split_features(config, args.split, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
