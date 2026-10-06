from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...data.collator import NLICollator
from ...data.config import DEFAULT_LABEL_MAPPING, DataLoaderConfig, DatasetConfig
from ...data.datamodule import build_dataloader
from ...data.dataset import NLIMultimodalDataset
from ...data.schemas import InputMode
from .config import PhoBERTXLSRConfig


def autocast_context(device: torch.device, enabled: bool) -> Any:
    """Use float16 autocast only when CUDA mixed precision is enabled."""

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        expected = ", ".join(mapping)
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of: {expected}.") from exc


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def build_raw_dataloader(
    config: PhoBERTXLSRConfig,
    split: str,
    *,
    batch_size: int,
) -> DataLoader:
    """Reuse source.data and optionally wrap item-level errors for skip policy."""

    dataset_config = DatasetConfig(
        data_path=config.data.split_path(split),
        input_mode=InputMode.TEXT_TEXT,
        label_mapping=DEFAULT_LABEL_MAPPING,
        validate_audio_exists=(
            config.data.validate_audio_exists and config.data.missing_audio_policy == "raise"
        ),
        allow_ambiguous_audio=config.data.allow_ambiguous_audio,
    )
    loader_config = DataLoaderConfig(
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )
    if config.data.missing_audio_policy == "raise":
        return build_dataloader(dataset_config, loader_config, shuffle=False)

    base_dataset = NLIMultimodalDataset(
        data_path=dataset_config.data_path,
        input_mode=dataset_config.input_mode,
        audio_root=dataset_config.audio_root,
        label_mapping=dataset_config.label_mapping,
        validate_audio_exists=False,
        allow_ambiguous_audio=dataset_config.allow_ambiguous_audio,
    )
    return DataLoader(
        _ErrorTolerantDataset(base_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=_ErrorTolerantCollator(DEFAULT_LABEL_MAPPING),
    )


class _ErrorTolerantDataset(Dataset):
    def __init__(self, dataset: NLIMultimodalDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            return self.dataset[index]
        except (FileNotFoundError, ValueError) as exc:
            record = self.dataset.records[index]
            return {
                "_dataset_error": {
                    "id": str(record.get("id")) if record.get("id") is not None else None,
                    "sample_index": index,
                    "stage": "data_loading",
                    "path": None,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            }


class _ErrorTolerantCollator:
    def __init__(self, label_mapping: dict[str, int]) -> None:
        self.collator = NLICollator(label_mapping)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [sample["_dataset_error"] for sample in samples if "_dataset_error" in sample]
        valid_samples = [sample for sample in samples if "_dataset_error" not in sample]
        batch = self.collator(valid_samples) if valid_samples else {"ids": [], "sample_indices": []}
        batch["errors"] = errors
        return batch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def reset_directory(path: str | Path, *, allowed_root: str | Path) -> Path:
    """Remove and recreate a pipeline-owned directory after a containment check."""

    directory = Path(path).resolve()
    root = Path(allowed_root).resolve()
    if directory == root or root not in directory.parents:
        raise ValueError(f"Refusing to reset directory outside the allowed root: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    return ensure_directory(directory)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    ensure_directory(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def torch_load(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def atomic_json_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    ensure_directory(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def append_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    if not records:
        return
    destination = Path(path)
    ensure_directory(destination.parent)
    with destination.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_sample_order_hash(
    sample_indices: list[int] | torch.Tensor,
    ids: list[str | None],
    labels: list[str | int],
) -> str:
    """Hash ordered sample identities so extraction and validation can detect drift."""

    indices = (
        sample_indices.detach().cpu().tolist()
        if isinstance(sample_indices, torch.Tensor)
        else list(sample_indices)
    )
    if not (len(indices) == len(ids) == len(labels)):
        raise ValueError("sample_indices, ids, and labels must have the same length.")
    digest = hashlib.sha256()
    for sample_index, sample_id, label in zip(indices, ids, labels):
        row = json.dumps(
            [int(sample_index), sample_id, label],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def peak_gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)
