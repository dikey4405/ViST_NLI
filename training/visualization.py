from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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

    return _extract_recorded_points(history, metric=metric, split=split)


def plot_loss_curves(
    history: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    run_id: str,
    best_epoch: int,
    logger: logging.Logger,
) -> list[Path]:
    """Plot recorded loss and epoch-end learning rate for one run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_history = [record for record in history if record.get("run_id") == run_id]
    epochs = sorted({record["epoch"] for record in run_history})
    lr_points = extract_learning_rate_points(run_history)
    artifacts: list[Path] = []
    for metric, title, filename in (
        ("total_loss", "Total Loss", "loss_total"),
        ("classification_loss", "Classification Loss", "loss_classification"),
    ):
        train_points = extract_loss_points(run_history, split="train", metric=metric)
        dev_points = extract_loss_points(run_history, split="dev", metric=metric)
        if not train_points and not dev_points:
            logger.warning("PLOT SKIPPED | %s has no values for run_id=%s", metric, run_id)
            continue
        panel_count = 2 if lr_points else 1
        if not lr_points:
            logger.warning("PLOT WARNING | no recorded learning rate for run_id=%s", run_id)
        fig, axes = plt.subplots(
            panel_count, 1, sharex=True, figsize=(7.4, 6.6 if lr_points else 4.6)
        )
        try:
            loss_axis = axes[0] if lr_points else axes
            _plot_loss_axis(
                loss_axis, epochs, train_points, dev_points,
                title=title, best_epoch=best_epoch, logger=logger,
            )
            if lr_points:
                _plot_lr_axis(axes[1], epochs, lr_points, best_epoch, logger)
            _finish_epoch_axes(axes[-1] if lr_points else axes, epochs)
            fig.tight_layout()
            path = output_dir / f"{filename}_{run_id}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            artifacts.append(path)
        finally:
            plt.close(fig)
    return artifacts


def plot_training_dynamics(
    history: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    run_id: str,
    best_epoch: int,
    logger: logging.Logger,
) -> Path | None:
    """Plot recorded LR, two losses, and Dev macro-F1 on aligned epoch axes."""

    run_history = [record for record in history if record.get("run_id") == run_id]
    epochs = sorted({record["epoch"] for record in run_history})
    if not epochs:
        logger.warning("PLOT SKIPPED | no epoch history for run_id=%s", run_id)
        return None

    lr_points = extract_learning_rate_points(run_history)
    f1_points = extract_dev_f1_points(run_history)
    loss_series = [
        (
            metric, title,
            extract_loss_points(run_history, split="train", metric=metric),
            extract_loss_points(run_history, split="dev", metric=metric),
        )
        for metric, title in (
            ("total_loss", "Total Loss"),
            ("classification_loss", "Classification Loss"),
        )
    ]
    panels: list[tuple[str, Any]] = []
    if lr_points:
        panels.append(("lr", lr_points))
    else:
        logger.warning("PLOT WARNING | no recorded learning rate for run_id=%s", run_id)
    for metric, title, train_points, dev_points in loss_series:
        if train_points or dev_points:
            panels.append(("loss", (title, train_points, dev_points)))
        else:
            logger.warning("PLOT WARNING | no %s values for run_id=%s", metric, run_id)
    if f1_points:
        panels.append(("f1", f1_points))
    else:
        logger.warning("PLOT WARNING | no Dev macro-F1 for run_id=%s", run_id)
    if not panels:
        return None

    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, figsize=(7.4, 2.4 * len(panels) + 1.0)
    )
    axes_list = list(axes) if len(panels) > 1 else [axes]
    try:
        for ax, (kind, values) in zip(axes_list, panels, strict=True):
            if kind == "lr":
                _plot_lr_axis(ax, epochs, values, best_epoch, logger)
            elif kind == "loss":
                title, train_points, dev_points = values
                _plot_loss_axis(
                    ax, epochs, train_points, dev_points,
                    title=title, best_epoch=best_epoch, logger=logger,
                )
            else:
                _plot_f1_axis(ax, epochs, values, best_epoch, logger)
        _finish_epoch_axes(axes_list[-1], epochs)
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"training_dynamics_{run_id}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        return path
    finally:
        plt.close(fig)


def extract_learning_rate_points(
    history: Sequence[Mapping[str, Any]],
) -> list[tuple[int, float]]:
    return _extract_recorded_points(history, metric="learning_rate")


def extract_dev_f1_points(
    history: Sequence[Mapping[str, Any]],
) -> list[tuple[int, float]]:
    return _extract_recorded_points(history, metric="macro_f1", split="dev", scale=100.0)


def _extract_recorded_points(
    history: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    split: str | None = None,
    scale: float = 1.0,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for record in history:
        values = record if split is None else record.get(split, {})
        if metric not in values:
            continue
        epoch, value = record["epoch"], values[metric]
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"Invalid epoch in plot history: {epoch!r}.")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {metric} at epoch {epoch}: {value!r}.")
        points.append((epoch, float(value) * scale))
    return points


def _plot_observations(
    ax: Axes, epochs: list[int], points: list[tuple[int, float]],
    *, color: str, linestyle: str, marker: str, label: str,
) -> None:
    values = dict(points)
    ax.plot(
        epochs,
        [values.get(epoch, float("nan")) for epoch in epochs],
        color=color, linestyle=linestyle, linewidth=1.9,
        marker=marker, markersize=3, label=label,
    )


def _mark_selected_epoch(ax: Axes, best_epoch: int) -> None:
    ax.axvline(
        best_epoch, color="#555555", linestyle="--", linewidth=1.2,
        label=f"Selected epoch ({best_epoch})",
    )


def _plot_loss_axis(
    ax: Axes, epochs: list[int], train_points: list[tuple[int, float]],
    dev_points: list[tuple[int, float]], *, title: str,
    best_epoch: int, logger: logging.Logger,
) -> None:
    if len(dev_points) < len(epochs):
        logger.warning(
            "PLOT WARNING | %s has %d Dev values for %d epochs; gaps are not imputed",
            title, len(dev_points), len(epochs),
        )
    if train_points:
        _plot_observations(
            ax, epochs, train_points, color="#176a83", linestyle="-",
            marker="o", label="Train",
        )
    if dev_points:
        _plot_observations(
            ax, epochs, dev_points, color="#b84a37", linestyle="--",
            marker="s", label="Dev",
        )
    _mark_selected_epoch(ax, best_epoch)
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def _plot_lr_axis(
    ax: Axes, epochs: list[int], points: list[tuple[int, float]],
    best_epoch: int, logger: logging.Logger,
) -> None:
    if len(points) < len(epochs):
        logger.warning(
            "PLOT WARNING | learning rate has %d values for %d epochs; gaps are not imputed",
            len(points), len(epochs),
        )
    _plot_observations(
        ax, epochs, points, color="#41664b", linestyle="-",
        marker="o", label="Epoch-end LR",
    )
    _mark_selected_epoch(ax, best_epoch)
    ax.set_title("Learning Rate Schedule")
    ax.set_ylabel("Learning rate (epoch end)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def _plot_f1_axis(
    ax: Axes, epochs: list[int], points: list[tuple[int, float]],
    best_epoch: int, logger: logging.Logger,
) -> None:
    if len(points) < len(epochs):
        logger.warning(
            "PLOT WARNING | Dev macro-F1 has %d values for %d epochs; gaps are not imputed",
            len(points), len(epochs),
        )
    _plot_observations(
        ax, epochs, points, color="#41664b", linestyle="-",
        marker="o", label="Dev macro-F1",
    )
    _mark_selected_epoch(ax, best_epoch)
    selected_value = dict(points).get(best_epoch)
    if selected_value is not None:
        ax.scatter(best_epoch, selected_value, color="#b84a37", s=38, zorder=4)
        ax.annotate(
            f"{selected_value:.1f}%", (best_epoch, selected_value),
            xytext=(6, 7), textcoords="offset points", fontsize=9,
        )
    ax.set_title("Dev Macro-F1")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def _finish_epoch_axes(ax: Axes, epochs: list[int]) -> None:
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.set_xlim(min(epochs) - 0.3, max(epochs) + 0.3)


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
