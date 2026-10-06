from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ...data.schemas import InputMode
from .cache_identity import compute_cache_fingerprint
from .cache_store import (
    build_cache_manifest,
    create_cache_manifest,
    load_manifest,
    modality_cache_directory,
    save_manifest,
    validate_manifest_shards,
    validate_resume_manifest,
)
from .config import PhoBERTXLSRConfig, load_pipeline_config
from .schemas import CacheManifest, CacheModality
from .speech_encoder import (
    AudioOverlengthError,
    AudioSkippedError,
    XLSRSpeechEncoder,
    load_audio,
)
from .text_encoder import PhoBERTTextEncoder
from .text_preprocessor import VietnameseTextPreprocessor
from .utils import (
    append_jsonl,
    atomic_torch_save,
    build_raw_dataloader,
    ensure_directory,
    get_logger,
    peak_gpu_memory_mb,
    reset_directory,
    resolve_device,
    resolve_torch_dtype,
    set_random_seed,
)


LOGGER = get_logger(__name__)


def cache_split_embeddings(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality | str,
) -> CacheManifest:
    """Cache one modality for one split using deterministic, resumable shards."""

    modality = CacheModality(modality)
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    storage_dtype = resolve_torch_dtype(config.cache.dtype)
    encoder_name = _encoder_name(config, modality)
    cache_directory = modality_cache_directory(config, split, modality)
    batch_size = (
        config.text_encoder.batch_size
        if modality == CacheModality.TEXT
        else config.speech_encoder.batch_size
    )
    dataloader = build_raw_dataloader(config, split, batch_size=batch_size)
    num_source_samples = len(dataloader.dataset)
    LOGGER.info("cache_fingerprint_start split=%s modality=%s", split, modality.value)
    cache_fingerprint = compute_cache_fingerprint(config, split, modality)

    if config.cache.overwrite and cache_directory.exists():
        reset_directory(cache_directory, allowed_root=config.cache.directory)
    else:
        ensure_directory(cache_directory)

    manifest = load_manifest(config, split, modality)
    error_report_path = (
        config.output.directory / "errors" / f"{split}_{modality.value}.jsonl"
    )
    if manifest is None:
        if error_report_path.exists():
            error_report_path.unlink()
        orphan_shards = sorted(cache_directory.glob("shard_*.pt"))
        if orphan_shards:
            raise RuntimeError(
                f"Cache shards exist without a manifest in {cache_directory}. "
                "Set cache.overwrite=true to rebuild this cache."
            )
        manifest = create_cache_manifest(
            config,
            split,
            modality,
            encoder_name,
            num_source_samples,
            cache_fingerprint=cache_fingerprint,
        )
        save_manifest(manifest, config, split, modality)
    else:
        validate_resume_manifest(
            manifest, config, split, modality, encoder_name,
            cache_fingerprint=cache_fingerprint,
        )
        validate_manifest_shards(cache_directory, manifest)
        if num_source_samples != manifest.num_source_samples:
            raise ValueError(
                f"Source sample count changed for split '{split}': "
                f"manifest={manifest.num_source_samples}, current={num_source_samples}."
            )
        if manifest.complete:
            LOGGER.info("cache_complete split=%s modality=%s; skipping", split, modality.value)
            return manifest

    LOGGER.info(
        "cache_start split=%s modality=%s samples=%d encoder=%s device=%s dtype=%s",
        split,
        modality.value,
        num_source_samples,
        encoder_name,
        device,
        config.cache.dtype,
    )
    encoder, preprocessor = _build_encoder(config, modality)
    processed = set(manifest.processed_sample_indices)
    skipped = set(manifest.skipped_sample_indices)
    shards = list(manifest.shards)
    rows: list[dict[str, Any]] = []
    truncated_audio_count = manifest.truncated_audio_count
    error_count = manifest.error_count
    started_at = time.perf_counter()

    progress = tqdm(total=num_source_samples, desc=f"cache {split}/{modality.value}", unit="sample")
    try:
        for batch_number, batch in enumerate(dataloader, start=1):
            batch_started_at = time.perf_counter()
            dataset_errors = _new_dataset_errors(batch, processed | skipped, split, modality)
            if dataset_errors:
                append_jsonl(dataset_errors, error_report_path)
                skipped.update(int(error["sample_index"]) for error in dataset_errors)
                error_count += len(dataset_errors)

            selected_positions = [
                position
                for position, sample_index in enumerate(batch.get("sample_indices", []))
                if int(sample_index) not in processed and int(sample_index) not in skipped
            ]
            if modality == CacheModality.TEXT:
                new_rows = _extract_text_rows(
                    batch,
                    selected_positions,
                    encoder,
                    preprocessor,
                    storage_dtype,
                )
                audio_errors: list[dict[str, Any]] = []
                batch_truncated = 0
            else:
                new_rows, audio_errors, batch_truncated = _extract_speech_rows(
                    batch,
                    selected_positions,
                    encoder,
                    config,
                    split,
                    storage_dtype,
                )
                if audio_errors:
                    append_jsonl(audio_errors, error_report_path)
                    skipped.update(int(error["sample_index"]) for error in audio_errors)
                    error_count += len(audio_errors)
                truncated_audio_count += batch_truncated

            rows.extend(new_rows)
            while len(rows) >= config.cache.shard_size:
                _save_row_shard(
                    rows,
                    config.cache.shard_size,
                    shards,
                    processed,
                    config,
                    split,
                    modality,
                    skipped,
                    encoder_name,
                    truncated_audio_count,
                    error_count,
                    num_source_samples,
                    cache_fingerprint,
                )

            scanned = len(batch.get("sample_indices", [])) + len(batch.get("errors", []))
            progress.update(scanned)
            LOGGER.info(
                "cache_batch split=%s modality=%s batch=%d encoded=%d skipped=%d "
                "elapsed_seconds=%.2f",
                split,
                modality.value,
                batch_number,
                len(new_rows),
                len(dataset_errors) + len(audio_errors),
                time.perf_counter() - batch_started_at,
            )

        if rows:
            _save_row_shard(
                rows,
                len(rows),
                shards,
                processed,
                config,
                split,
                modality,
                skipped,
                encoder_name,
                truncated_audio_count,
                error_count,
                num_source_samples,
                cache_fingerprint,
            )

        if len(processed | skipped) != num_source_samples:
            missing = sorted(set(range(num_source_samples)) - processed - skipped)
            raise RuntimeError(
                f"Cache did not account for every source sample in split '{split}'; "
                f"missing sample_index values: {missing[:10]}."
            )
        manifest = build_cache_manifest(
            config,
            split,
            modality,
            encoder_name,
            shards=shards,
            processed=processed,
            skipped=skipped,
            truncated_audio_count=truncated_audio_count,
            error_count=error_count,
            num_source_samples=num_source_samples,
            complete=True,
            cache_fingerprint=cache_fingerprint,
        )
        save_manifest(manifest, config, split, modality)
    finally:
        progress.close()
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    LOGGER.info(
        "cache_done split=%s modality=%s processed=%d skipped=%d truncated_audio=%d "
        "elapsed_seconds=%.2f peak_gpu_memory_mb=%s",
        split,
        modality.value,
        len(processed),
        len(skipped),
        truncated_audio_count,
        time.perf_counter() - started_at,
        peak_gpu_memory_mb(device),
    )
    return manifest


def _build_encoder(
    config: PhoBERTXLSRConfig,
    modality: CacheModality,
) -> tuple[PhoBERTTextEncoder | XLSRSpeechEncoder, VietnameseTextPreprocessor | None]:
    if modality == CacheModality.TEXT:
        encoder = PhoBERTTextEncoder(
            model_name=config.text_encoder.model_name,
            max_length=config.text_encoder.max_length,
            expected_output_dim=config.projector.input_dim,
            freeze=config.text_encoder.freeze,
            device=config.device,
            mixed_precision=config.mixed_precision,
        )
        preprocessor = VietnameseTextPreprocessor(
            use_word_segmentation=config.text_encoder.use_word_segmentation,
            cache_segmented_text=config.text_encoder.cache_segmented_text,
            cache_path=config.cache.directory / "segmented_text.json",
        )
        return encoder, preprocessor

    encoder = XLSRSpeechEncoder(
        model_name=config.speech_encoder.model_name,
        target_sample_rate=config.speech_encoder.target_sample_rate,
        max_audio_seconds=config.speech_encoder.max_audio_seconds,
        overlength_policy=config.speech_encoder.overlength_policy,
        expected_output_dim=config.projector.input_dim,
        freeze=config.speech_encoder.freeze,
        device=config.device,
        mixed_precision=config.mixed_precision,
    )
    return encoder, None


def _extract_text_rows(
    batch: dict[str, Any],
    positions: list[int],
    encoder: PhoBERTTextEncoder | XLSRSpeechEncoder,
    preprocessor: VietnameseTextPreprocessor | None,
    storage_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if not positions:
        return []
    if not isinstance(encoder, PhoBERTTextEncoder) or preprocessor is None:
        raise TypeError("Text cache extraction requires PhoBERTTextEncoder and a preprocessor.")
    pair = batch["input_pairs"][InputMode.TEXT_TEXT.value]
    premise_texts = [str(pair["premise"][position]) for position in positions]
    hypothesis_texts = [str(pair["hypothesis"][position]) for position in positions]
    processed_texts = preprocessor.preprocess_many(premise_texts + hypothesis_texts)
    split_at = len(premise_texts)
    premise_embeddings = encoder.encode(processed_texts[:split_at])
    hypothesis_embeddings = encoder.encode(processed_texts[split_at:])
    return _build_rows(
        batch,
        positions,
        premise_embeddings,
        hypothesis_embeddings,
        storage_dtype,
    )


def _extract_speech_rows(
    batch: dict[str, Any],
    positions: list[int],
    encoder: PhoBERTTextEncoder | XLSRSpeechEncoder,
    config: PhoBERTXLSRConfig,
    split: str,
    storage_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not positions:
        return [], [], 0
    if not isinstance(encoder, XLSRSpeechEncoder):
        raise TypeError("Speech cache extraction requires XLSRSpeechEncoder.")

    pair = batch["input_pairs"][InputMode.SPEECH_SPEECH.value]
    valid_positions: list[int] = []
    premise_waveforms: list[torch.Tensor] = []
    hypothesis_waveforms: list[torch.Tensor] = []
    errors: list[dict[str, Any]] = []
    truncated_audio_count = 0

    for position in positions:
        sample_id = batch["ids"][position]
        sample_index = int(batch["sample_indices"][position])
        premise_path = str(pair["premise"][position])
        hypothesis_path = str(pair["hypothesis"][position])
        try:
            premise_audio = _load_configured_audio(premise_path, config)
            hypothesis_audio = _load_configured_audio(hypothesis_path, config)
        except AudioOverlengthError as exc:
            LOGGER.error(
                "audio_overlength split=%s id=%s sample_index=%d message=%s",
                split,
                sample_id,
                sample_index,
                exc,
            )
            raise
        except (AudioSkippedError, FileNotFoundError, RuntimeError, ValueError) as exc:
            should_raise = (
                not isinstance(exc, AudioSkippedError)
                and config.data.missing_audio_policy == "raise"
            )
            if should_raise:
                raise RuntimeError(
                    f"Speech preprocessing failed for split='{split}', id='{sample_id}', "
                    f"sample_index={sample_index}: {exc}"
                ) from exc
            errors.append(
                {
                    "id": sample_id,
                    "sample_index": sample_index,
                    "split": split,
                    "modality": CacheModality.SPEECH.value,
                    "stage": "speech_preprocessing",
                    "path": _error_path(exc, premise_path, hypothesis_path),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        for side, path, loaded in (
            ("premise", premise_path, premise_audio),
            ("hypothesis", hypothesis_path, hypothesis_audio),
        ):
            if loaded.was_truncated:
                truncated_audio_count += 1
                LOGGER.warning(
                    "audio_truncated split=%s id=%s sample_index=%d side=%s path=%s",
                    split,
                    sample_id,
                    sample_index,
                    side,
                    path,
                )
        valid_positions.append(position)
        premise_waveforms.append(premise_audio.waveform)
        hypothesis_waveforms.append(hypothesis_audio.waveform)

    if not valid_positions:
        return [], errors, truncated_audio_count
    premise_embeddings = encoder.encode_waveforms(premise_waveforms)
    hypothesis_embeddings = encoder.encode_waveforms(hypothesis_waveforms)
    rows = _build_rows(
        batch,
        valid_positions,
        premise_embeddings,
        hypothesis_embeddings,
        storage_dtype,
    )
    return rows, errors, truncated_audio_count


def _load_configured_audio(path: str, config: PhoBERTXLSRConfig) -> Any:
    return load_audio(
        path,
        target_sample_rate=config.speech_encoder.target_sample_rate,
        max_audio_seconds=config.speech_encoder.max_audio_seconds,
        overlength_policy=config.speech_encoder.overlength_policy,
    )


def _build_rows(
    batch: dict[str, Any],
    positions: list[int],
    premise_embeddings: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
    storage_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    expected_shape = (len(positions), 1024)
    if premise_embeddings.shape != expected_shape or hypothesis_embeddings.shape != expected_shape:
        raise ValueError(
            f"Encoder output must have shape {expected_shape}; got "
            f"{tuple(premise_embeddings.shape)} and {tuple(hypothesis_embeddings.shape)}."
        )
    premise_embeddings = premise_embeddings.detach().cpu().to(storage_dtype)
    hypothesis_embeddings = hypothesis_embeddings.detach().cpu().to(storage_dtype)
    return [
        {
            "id": batch["ids"][position],
            "label": batch["labels"][position],
            "sample_index": int(batch["sample_indices"][position]),
            "premise_embedding": premise_embeddings[row_index],
            "hypothesis_embedding": hypothesis_embeddings[row_index],
        }
        for row_index, position in enumerate(positions)
    ]


def _save_row_shard(
    rows: list[dict[str, Any]],
    count: int,
    shards: list[str],
    processed: set[int],
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
    skipped: set[int],
    encoder_name: str,
    truncated_audio_count: int,
    error_count: int,
    num_source_samples: int,
    cache_fingerprint: str,
) -> None:
    shard_rows = rows[:count]
    del rows[:count]
    shard_name = f"shard_{len(shards):05d}.pt"
    shard_path = modality_cache_directory(config, split, modality) / shard_name
    payload = {
        "ids": [row["id"] for row in shard_rows],
        "labels": [row["label"] for row in shard_rows],
        "sample_indices": torch.tensor(
            [row["sample_index"] for row in shard_rows], dtype=torch.long
        ),
        "premise_embeddings": torch.stack(
            [row["premise_embedding"] for row in shard_rows]
        ),
        "hypothesis_embeddings": torch.stack(
            [row["hypothesis_embedding"] for row in shard_rows]
        ),
    }
    atomic_torch_save(payload, shard_path)
    shards.append(shard_name)
    processed.update(int(row["sample_index"]) for row in shard_rows)
    manifest = build_cache_manifest(
        config,
        split,
        modality,
        encoder_name,
        shards=shards,
        processed=processed,
        skipped=skipped,
        truncated_audio_count=truncated_audio_count,
        error_count=error_count,
        num_source_samples=num_source_samples,
        complete=False,
        cache_fingerprint=cache_fingerprint,
    )
    save_manifest(manifest, config, split, modality)
    LOGGER.info("cache_shard_saved path=%s rows=%d", shard_path, len(shard_rows))


def _new_dataset_errors(
    batch: dict[str, Any],
    completed_indices: set[int],
    split: str,
    modality: CacheModality,
) -> list[dict[str, Any]]:
    errors = []
    for raw_error in batch.get("errors", []):
        sample_index = int(raw_error["sample_index"])
        if sample_index in completed_indices:
            continue
        error = dict(raw_error)
        error.update({"split": split, "modality": modality.value})
        errors.append(error)
    return errors


def _error_path(exc: Exception, premise_path: str, hypothesis_path: str) -> str:
    message = str(exc)
    if hypothesis_path in message:
        return hypothesis_path
    return premise_path


def _encoder_name(config: PhoBERTXLSRConfig, modality: CacheModality) -> str:
    if modality == CacheModality.TEXT:
        return config.text_encoder.model_name
    return config.speech_encoder.model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen PhoBERT or XLS-R embeddings.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument(
        "--modality",
        choices=(CacheModality.TEXT.value, CacheModality.SPEECH.value, "all"),
        default="all",
        help="Cache text, speech, or both sequentially. 'all' never keeps both encoders loaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    modalities = (
        (CacheModality.TEXT, CacheModality.SPEECH)
        if args.modality == "all"
        else (CacheModality(args.modality),)
    )
    for modality in modalities:
        cache_split_embeddings(config, args.split, modality)


if __name__ == "__main__":
    main()
