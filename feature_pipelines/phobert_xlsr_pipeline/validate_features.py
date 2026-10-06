from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ...data.config import DEFAULT_LABEL_MAPPING
from .config import PhoBERTXLSRConfig, load_pipeline_config
from .schemas import MODE_ORDER, PAIR_FEATURE_COMPONENTS
from .utils import (
    atomic_json_save,
    compute_sample_order_hash,
    get_logger,
    torch_load,
)


LOGGER = get_logger(__name__)
REQUIRED_SPLITS = ("train", "dev", "test")


def validate_feature_outputs(
    config: PhoBERTXLSRConfig,
    splits: tuple[str, ...] = REQUIRED_SPLITS,
) -> dict[str, Any]:
    """Validate saved feature files and always persist a machine-readable report."""

    errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, dict[str, Any]] = {}
    metadata = _load_metadata(config.output.directory / "metadata.json", errors)
    _validate_metadata_contract(metadata, config, errors)

    all_ids: dict[str, list[str]] = {}
    for split in splits:
        split_report, split_ids = _validate_split(config, split, metadata, errors, warnings)
        split_reports[split] = split_report
        all_ids[split] = split_ids

    source_paths = [metadata.get("split_source_files", {}).get(split) for split in splits]
    output_paths = [metadata.get("split_output_files", {}).get(split) for split in splits]
    if None not in source_paths and len(set(source_paths)) != len(source_paths):
        errors.append("Metadata maps multiple splits to the same source data file.")
    if None not in output_paths and len(set(output_paths)) != len(output_paths):
        errors.append("Metadata maps multiple splits to the same output feature file.")

    cross_split_duplicates = _cross_split_duplicate_ids(all_ids)
    if cross_split_duplicates:
        warnings.append(
            f"Raw ids overlap across splits ({len(cross_split_duplicates)} ids). "
            "Split plus sample_index remains the storage identity."
        )

    report = {
        "status": "passed" if not errors else "failed",
        "feature_source": config.pipeline_name,
        "identity_field": "sample_index",
        "required_mode_order": list(MODE_ORDER),
        "required_pair_feature_shape": [4, config.pair_feature.output_dim],
        "splits": split_reports,
        "cross_split_duplicate_raw_id_count": len(cross_split_duplicates),
        "errors": errors,
        "warnings": warnings,
    }
    report_path = config.output.directory / "validation_report.json"
    atomic_json_save(report, report_path)
    LOGGER.info("validation_report=%s status=%s", report_path, report["status"])
    if errors:
        raise ValueError(
            f"Feature validation failed with {len(errors)} error(s). See {report_path}."
        )
    return report


def _validate_split(
    config: PhoBERTXLSRConfig,
    split: str,
    metadata: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    output_path = config.output.directory / f"{split}.pt"
    report: dict[str, Any] = {
        "path": str(output_path.resolve()),
        "exists": output_path.exists(),
    }
    if not output_path.exists():
        errors.append(f"Missing feature file for split '{split}': {output_path}")
        return report, []

    try:
        payload = torch_load(output_path)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        errors.append(f"Cannot load feature file for split '{split}': {exc}")
        return report, []
    if not isinstance(payload, list):
        errors.append(f"Feature file for split '{split}' must contain a list of samples.")
        return report, []

    ids: list[str] = []
    labels: list[str | int] = []
    sample_indices: list[int] = []
    invalid_shape_count = 0
    non_finite_count = 0
    invalid_mode_count = 0
    missing_id_count = 0
    invalid_label_count = 0

    for row_index, sample in enumerate(payload):
        context = f"split='{split}', row={row_index}"
        if not isinstance(sample, dict):
            errors.append(f"Sample must be a dictionary at {context}.")
            continue
        sample_id = sample.get("id")
        if sample_id is None or str(sample_id).strip() == "":
            missing_id_count += 1
        else:
            ids.append(str(sample_id))
        label = sample.get("label")
        labels.append(label)
        if not _is_valid_label(label):
            invalid_label_count += 1
        sample_index = sample.get("sample_index")
        if not isinstance(sample_index, int):
            errors.append(f"sample_index must be an int at {context}.")
        else:
            sample_indices.append(sample_index)

        features = sample.get("features")
        if not isinstance(features, dict) or tuple(features.keys()) != MODE_ORDER:
            invalid_mode_count += 1
            continue
        for mode in MODE_ORDER:
            feature = features[mode]
            if not isinstance(feature, torch.Tensor) or feature.shape != (
                config.pair_feature.output_dim,
            ):
                invalid_shape_count += 1
                continue
            if not torch.isfinite(feature).all():
                non_finite_count += 1

    duplicate_indices = len(sample_indices) - len(set(sample_indices))
    duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
    if len(sample_indices) != len(payload):
        errors.append(f"Some samples in split '{split}' do not have a valid sample_index.")
    if duplicate_indices:
        errors.append(
            f"Split '{split}' contains {duplicate_indices} duplicate sample_index values."
        )
    if sample_indices != sorted(sample_indices):
        errors.append(f"Split '{split}' is not ordered deterministically by sample_index.")
    if missing_id_count:
        errors.append(f"Split '{split}' contains {missing_id_count} missing ids.")
    if invalid_label_count:
        errors.append(f"Split '{split}' contains {invalid_label_count} invalid labels.")
    if invalid_mode_count:
        errors.append(
            f"Split '{split}' contains {invalid_mode_count} samples with missing or "
            "misordered modes."
        )
    if invalid_shape_count:
        errors.append(
            f"Split '{split}' contains {invalid_shape_count} mode features with invalid shape."
        )
    if non_finite_count:
        errors.append(f"Split '{split}' contains {non_finite_count} non-finite mode features.")
    if duplicate_ids:
        warnings.append(
            f"Split '{split}' contains {duplicate_ids} repeated raw ids from the source data; "
            "sample_index values are unique."
        )

    _validate_split_metadata(
        config,
        split,
        output_path,
        metadata,
        sample_indices,
        ids,
        labels,
        errors,
    )
    report.update(
        {
            "num_samples": len(payload),
            "logical_shape": [len(payload), len(MODE_ORDER), config.pair_feature.output_dim],
            "duplicate_sample_index_count": duplicate_indices,
            "duplicate_raw_id_count": duplicate_ids,
            "missing_id_count": missing_id_count,
            "invalid_label_count": invalid_label_count,
            "invalid_mode_count": invalid_mode_count,
            "invalid_shape_count": invalid_shape_count,
            "non_finite_count": non_finite_count,
            "skipped_samples": metadata.get("skipped_counts", {}).get(split),
            "cache_statistics": metadata.get("cache_statistics", {}).get(split),
        }
    )
    return report, ids


def _validate_metadata_contract(
    metadata: dict[str, Any],
    config: PhoBERTXLSRConfig,
    errors: list[str],
) -> None:
    expected = {
        "feature_source": config.pipeline_name,
        "text_encoder": config.text_encoder.model_name,
        "speech_encoder": config.speech_encoder.model_name,
        "text_pooling": config.text_encoder.pooling,
        "speech_pooling": config.speech_encoder.pooling,
        "embedding_dim": config.projector.output_dim,
        "pair_feature_dim": config.pair_feature.output_dim,
        "pair_feature_components": list(PAIR_FEATURE_COMPONENTS),
        "mode_order": list(MODE_ORDER),
        "label_mapping": DEFAULT_LABEL_MAPPING,
        "identity_field": "sample_index",
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            errors.append(
                f"Metadata field '{key}' must equal {expected_value!r}, "
                f"got {metadata.get(key)!r}."
            )
    checkpoint = metadata.get("projector_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        errors.append("Metadata does not record projector_checkpoint.")
    elif not Path(checkpoint).exists():
        errors.append(f"Projector checkpoint recorded in metadata does not exist: {checkpoint}")


def _validate_split_metadata(
    config: PhoBERTXLSRConfig,
    split: str,
    output_path: Path,
    metadata: dict[str, Any],
    sample_indices: list[int],
    ids: list[str],
    labels: list[str | int],
    errors: list[str],
) -> None:
    if metadata.get("split_counts", {}).get(split) != len(sample_indices):
        errors.append(f"Metadata sample count does not match output for split '{split}'.")
    expected_source = str(config.data.split_path(split).resolve())
    if metadata.get("split_source_files", {}).get(split) != expected_source:
        errors.append(f"Metadata source file does not match config for split '{split}'.")
    if metadata.get("split_output_files", {}).get(split) != str(output_path.resolve()):
        errors.append(f"Metadata output file does not match actual path for split '{split}'.")
    if len(sample_indices) == len(ids) == len(labels):
        order_hash = compute_sample_order_hash(sample_indices, ids, labels)
        if metadata.get("split_order_hashes", {}).get(split) != order_hash:
            errors.append(f"Deterministic sample-order hash mismatch for split '{split}'.")


def _load_metadata(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"Missing metadata file: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read metadata file {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"Metadata file must contain a JSON object: {path}")
        return {}
    return payload


def _is_valid_label(label: Any) -> bool:
    if isinstance(label, str):
        return label in DEFAULT_LABEL_MAPPING
    if isinstance(label, int) and not isinstance(label, bool):
        return label in DEFAULT_LABEL_MAPPING.values()
    return False


def _cross_split_duplicate_ids(ids_by_split: dict[str, list[str]]) -> set[str]:
    owners: dict[str, set[str]] = {}
    for split, ids in ids_by_split.items():
        for sample_id in ids:
            owners.setdefault(sample_id, set()).add(split)
    return {sample_id for sample_id, splits in owners.items() if len(splits) > 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PhoBERT/XLS-R feature outputs.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_feature_outputs(load_pipeline_config(args.config))


if __name__ == "__main__":
    main()
