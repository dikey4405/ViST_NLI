from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from ..data.config import DEFAULT_LABEL_MAPPING


def extract_loss_points(
    history: Sequence[Mapping[str, Any]],
    *,
    split: str,
    metric: str,
) -> list[tuple[int, float]]:
    """Return recorded epoch/value pairs without filling missing observations."""

    points: list[tuple[int, float]] = []
    for record in history:
        split_metrics = record.get(split, {})
        if metric not in split_metrics:
            continue
        epoch = record["epoch"]
        value = split_metrics[metric]
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"Invalid epoch in plot history: {epoch!r}.")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(
                f"Invalid {split} {metric} at epoch {epoch}: {value!r}."
            )
        points.append((epoch, float(value)))
    return points


def plot_loss_curves(
    history: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    run_id: str,
    best_epoch: int,
    logger: logging.Logger,
) -> list[Path]:
    """Plot the two losses already reported by the trainer for one run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_history = [record for record in history if record.get("run_id") == run_id]
    artifacts: list[Path] = []
    for metric, title, filename in (
        ("total_loss", "Total Loss", "loss_total"),
        ("classification_loss", "Classification Loss", "loss_classification"),
    ):
        train_points = extract_loss_points(run_history, split="train", metric=metric)
        dev_points = extract_loss_points(run_history, split="dev", metric=metric)
        if not train_points:
            logger.warning("PLOT SKIPPED | %s has no Train values for run_id=%s", metric, run_id)
            continue
        if len(dev_points) < len(train_points):
            logger.warning(
                "PLOT WARNING | %s has %d Dev values for %d Train epochs; "
                "missing Dev values are not imputed",
                metric,
                len(dev_points),
                len(train_points),
            )

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        try:
            train_epochs = [epoch for epoch, _ in train_points]
            ax.plot(
                train_epochs,
                [value for _, value in train_points],
                color="#176a83",
                linestyle="-",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Train",
            )
            if dev_points:
                dev_by_epoch = dict(dev_points)
                ax.plot(
                    train_epochs,
                    [dev_by_epoch.get(epoch, float("nan")) for epoch in train_epochs],
                    color="#b84a37",
                    linestyle="--",
                    linewidth=2,
                    marker="s",
                    markersize=3,
                    label="Dev",
                )
            ax.axvline(
                best_epoch,
                color="#555555",
                linestyle=":",
                linewidth=1.3,
                label=f"Selected epoch ({best_epoch})",
            )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.tight_layout()
            path = output_dir / f"{filename}_{run_id}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            artifacts.append(path)
        finally:
            plt.close(fig)
    return artifacts


def save_confusion_artifacts(
    confusion: torch.Tensor,
    *,
    split: str,
    output_dir: Path,
    run_id: str,
) -> list[Path]:
    """Save counts, row percentages, and a count CSV in label-ID order."""

    if split not in {"dev", "test"}:
        raise ValueError(f"Confusion split must be 'dev' or 'test', got {split!r}.")
    label_items = sorted(DEFAULT_LABEL_MAPPING.items(), key=lambda item: item[1])
    label_ids = [index for _, index in label_items]
    if label_ids != list(range(len(label_items))):
        raise ValueError("NLI label IDs must be contiguous from zero for plotting.")
    labels = [name.capitalize() for name, _ in label_items]
    if confusion.ndim != 2 or tuple(confusion.shape) != (len(labels), len(labels)):
        raise ValueError(
            f"Confusion must have shape [{len(labels)}, {len(labels)}], "
            f"got {tuple(confusion.shape)}."
        )
    if confusion.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Confusion must contain integer counts.")
    if (confusion < 0).any():
        raise ValueError("Confusion counts must be non-negative.")

    counts = confusion.detach().cpu().numpy().astype(np.int64, copy=False)
    percentages = normalize_confusion_rows(counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    csv_path = output_dir / f"confusion_matrix_{split}_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label/predicted_label", *labels])
        for label, row in zip(labels, counts, strict=True):
            writer.writerow([label, *(int(value) for value in row)])
    artifacts.append(csv_path)

    for normalized, matrix, suffix, colorbar_label in (
        (False, counts, "", "Samples"),
        (True, percentages, "_normalized", "Percentage (%)"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        try:
            image = ax.imshow(
                matrix,
                cmap="Blues",
                vmin=0,
                vmax=100 if normalized else max(int(counts.max()), 1),
            )
            fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
            for true_index in range(len(labels)):
                for predicted_index in range(len(labels)):
                    value = matrix[true_index, predicted_index]
                    label = f"{value:.1f}%" if normalized else str(int(value))
                    midpoint = 50 if normalized else max(int(counts.max()), 1) / 2
                    ax.text(
                        predicted_index,
                        true_index,
                        label,
                        ha="center",
                        va="center",
                        color="white" if value > midpoint else "#1b1b1b",
                    )
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            title_suffix = " (Row %)" if normalized else " (Counts)"
            ax.set_title(f"{split.capitalize()} Confusion Matrix{title_suffix}")
            fig.tight_layout()
            path = output_dir / f"confusion_matrix_{split}{suffix}_{run_id}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            artifacts.append(path)
        finally:
            plt.close(fig)
    return artifacts


def normalize_confusion_rows(counts: np.ndarray) -> np.ndarray:
    """Convert integer counts to percentages, leaving empty rows at zero."""

    row_totals = counts.sum(axis=1, keepdims=True)
    return counts.astype(np.float64) / np.maximum(row_totals, 1) * 100.0
