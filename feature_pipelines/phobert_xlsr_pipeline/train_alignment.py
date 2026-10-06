from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .alignment_loss import SymmetricTextSpeechAlignmentLoss, compute_alignment_metrics
from .cache_store import load_merged_cached_embeddings
from .config import PhoBERTXLSRConfig, load_pipeline_config
from .projectors import SpeechProjector, TextProjector, build_projector_pair
from .schemas import BEST_PROJECTOR_CHECKPOINT_NAME
from .utils import (
    atomic_json_save,
    atomic_torch_save,
    ensure_directory,
    get_logger,
    peak_gpu_memory_mb,
    resolve_device,
    set_random_seed,
)


LOGGER = get_logger(__name__)


class CachedAlignmentDataset(Dataset):
    """Dataset over frozen base embeddings; no base encoder is loaded here."""

    EMBEDDING_KEYS = (
        "premise_text_embeddings",
        "premise_speech_embeddings",
        "hypothesis_text_embeddings",
        "hypothesis_speech_embeddings",
    )

    def __init__(self, cache: dict[str, Any]) -> None:
        self.sample_indices = cache["sample_indices"].long()
        self.embeddings = {key: cache[key] for key in self.EMBEDDING_KEYS}
        self.sentence_groups: dict[str, torch.Tensor] = {}
        for side in ("premise", "hypothesis"):
            keys = cache.get(f"{side}_sentence_keys")
            if keys is None or len(keys) != len(self.sample_indices):
                raise ValueError(
                    f"Cache requires {side}_sentence_keys for duplicate-aware alignment."
                )
            groups = {key: index for index, key in enumerate(sorted(set(keys)))}
            self.sentence_groups[f"{side}_sentence_group"] = torch.tensor(
                [groups[key] for key in keys], dtype=torch.long
            )
        self._validate()

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sample_index": self.sample_indices[index],
            **{key: values[index] for key, values in self.embeddings.items()},
            **{key: values[index] for key, values in self.sentence_groups.items()},
        }

    def _validate(self) -> None:
        num_samples = len(self)
        for key, values in self.embeddings.items():
            if values.shape != (num_samples, 1024):
                raise ValueError(
                    f"{key} must have shape {(num_samples, 1024)}, got {tuple(values.shape)}."
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"{key} contains NaN or Inf values.")


def train_alignment(config: PhoBERTXLSRConfig) -> Path:
    """Train only modality-specific projectors on cached sentence embeddings."""

    if not config.alignment.enabled:
        raise ValueError("Alignment training is disabled in the pipeline config.")
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    train_dataset = CachedAlignmentDataset(load_merged_cached_embeddings(config, "train"))
    dev_dataset = CachedAlignmentDataset(load_merged_cached_embeddings(config, "dev"))
    if len(train_dataset) < 2 or len(dev_dataset) < 2:
        raise ValueError("Train and dev alignment caches must each contain at least two samples.")

    train_loader = _build_alignment_loader(train_dataset, config, training=True)
    dev_loader = _build_alignment_loader(dev_dataset, config, training=False)
    text_projector, speech_projector = _build_projectors(config, device)
    criterion = SymmetricTextSpeechAlignmentLoss(config.alignment.temperature)
    optimizer = torch.optim.AdamW(
        list(text_projector.parameters()) + list(speech_projector.parameters()),
        lr=config.alignment.learning_rate,
        weight_decay=config.alignment.weight_decay,
    )

    checkpoint_directory = ensure_directory(config.checkpoint.directory)
    checkpoint_path = checkpoint_directory / BEST_PROJECTOR_CHECKPOINT_NAME
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    started_at = time.perf_counter()
    LOGGER.info(
        "alignment_start train_samples=%d dev_samples=%d device=%s temperature=%.4f",
        len(train_dataset),
        len(dev_dataset),
        device,
        config.alignment.temperature,
    )

    for epoch in range(1, config.alignment.epochs + 1):
        train_metrics = _run_epoch(
            train_loader,
            text_projector,
            speech_projector,
            criterion,
            device,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
            dev_loader,
            text_projector,
            speech_projector,
            criterion,
            device,
            optimizer=None,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        LOGGER.info(
            "alignment_epoch epoch=%d train_loss=%.6f validation_loss=%.6f "
            "validation_t2s_accuracy=%.4f validation_s2t_accuracy=%.4f "
            "positive_cosine=%.4f negative_cosine=%.4f",
            epoch,
            train_metrics["loss"],
            validation_metrics["loss"],
            validation_metrics["text_to_speech_accuracy"],
            validation_metrics["speech_to_text_accuracy"],
            validation_metrics["positive_cosine_similarity"],
            validation_metrics["negative_cosine_similarity"],
        )

        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            _save_projector_checkpoint(
                checkpoint_path,
                text_projector,
                speech_projector,
                config,
                epoch,
                best_validation_loss,
                validation_metrics,
            )
        else:
            epochs_without_improvement += 1

        if not config.checkpoint.save_best_only:
            _save_projector_checkpoint(
                checkpoint_directory / "last_projectors.pt",
                text_projector,
                speech_projector,
                config,
                epoch,
                best_validation_loss,
                validation_metrics,
            )
        if epochs_without_improvement >= config.alignment.patience:
            LOGGER.info("alignment_early_stopping epoch=%d", epoch)
            break

    atomic_json_save(
        {
            "best_validation_loss": best_validation_loss,
            "checkpoint": str(checkpoint_path.resolve()),
            "history": history,
        },
        checkpoint_directory / "alignment_history.json",
    )
    LOGGER.info(
        "alignment_done checkpoint=%s elapsed_seconds=%.2f peak_gpu_memory_mb=%s",
        checkpoint_path,
        time.perf_counter() - started_at,
        peak_gpu_memory_mb(device),
    )
    return checkpoint_path


def _build_projectors(
    config: PhoBERTXLSRConfig,
    device: torch.device,
) -> tuple[TextProjector, SpeechProjector]:
    return build_projector_pair(
        input_dim=config.projector.input_dim,
        hidden_dim=config.projector.hidden_dim,
        output_dim=config.projector.output_dim,
        dropout=config.projector.dropout,
        device=device,
    )


def _build_alignment_loader(
    dataset: CachedAlignmentDataset,
    config: PhoBERTXLSRConfig,
    *,
    training: bool,
) -> DataLoader:
    batch_size = min(config.alignment.batch_size, len(dataset))
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        generator=generator if training else None,
    )


def _run_epoch(
    loader: DataLoader,
    text_projector: TextProjector,
    speech_projector: SpeechProjector,
    criterion: SymmetricTextSpeechAlignmentLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    text_projector.train(training)
    speech_projector.train(training)
    totals = {
        "loss": 0.0,
        "text_to_speech_accuracy": 0.0,
        "speech_to_text_accuracy": 0.0,
        "positive_cosine_similarity": 0.0,
        "negative_cosine_similarity": 0.0,
    }
    num_samples = 0

    for batch in loader:
        premise_text = batch["premise_text_embeddings"].to(device=device, dtype=torch.float32)
        premise_speech = batch["premise_speech_embeddings"].to(
            device=device, dtype=torch.float32
        )
        hypothesis_text = batch["hypothesis_text_embeddings"].to(
            device=device, dtype=torch.float32
        )
        hypothesis_speech = batch["hypothesis_speech_embeddings"].to(
            device=device, dtype=torch.float32
        )
        premise_groups = batch["premise_sentence_group"].to(device)
        hypothesis_groups = batch["hypothesis_sentence_group"].to(device)
        premise_positives = premise_groups[:, None].eq(premise_groups[None, :])
        hypothesis_positives = hypothesis_groups[:, None].eq(hypothesis_groups[None, :])
        with torch.set_grad_enabled(training):
            projected_premise_text = text_projector(premise_text)
            projected_premise_speech = speech_projector(premise_speech)
            projected_hypothesis_text = text_projector(hypothesis_text)
            projected_hypothesis_speech = speech_projector(hypothesis_speech)
            premise_loss = criterion(
                projected_premise_text, projected_premise_speech, premise_positives
            )
            hypothesis_loss = criterion(
                projected_hypothesis_text, projected_hypothesis_speech, hypothesis_positives
            )
            loss = 0.5 * (premise_loss + hypothesis_loss)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            premise_metrics = compute_alignment_metrics(
                projected_premise_text, projected_premise_speech, premise_positives
            )
            hypothesis_metrics = compute_alignment_metrics(
                projected_hypothesis_text, projected_hypothesis_speech, hypothesis_positives
            )
        batch_size = int(premise_text.shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for key in premise_metrics:
            value = 0.5 * (premise_metrics[key] + hypothesis_metrics[key])
            totals[key] += float(value.detach()) * batch_size
        num_samples += batch_size

    if num_samples == 0:
        raise RuntimeError("Alignment DataLoader produced no batches.")
    return {key: value / num_samples for key, value in totals.items()}


def _save_projector_checkpoint(
    path: Path,
    text_projector: nn.Module,
    speech_projector: nn.Module,
    config: PhoBERTXLSRConfig,
    epoch: int,
    best_validation_loss: float,
    validation_metrics: dict[str, float],
) -> None:
    payload = {
        "text_projector_state_dict": _cpu_state_dict(text_projector),
        "speech_projector_state_dict": _cpu_state_dict(speech_projector),
        "config": config.to_dict(),
        "best_validation_loss": best_validation_loss,
        "validation_metrics": validation_metrics,
        "epoch": epoch,
    }
    atomic_torch_save(payload, path)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhoBERT/XLS-R alignment projectors.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_alignment(load_pipeline_config(args.config))


if __name__ == "__main__":
    main()
