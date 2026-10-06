from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .metrics import MODE_ABBREVIATIONS


METRIC_PRECISION = 6


def create_run_id() -> str:
    """Return a sortable UTC identifier for one training invocation."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def format_epoch_summary(
    *,
    epoch: int,
    num_epochs: int,
    epoch_time_seconds: float,
    learning_rate: float,
    train_metrics: dict[str, float],
    dev_metrics: dict[str, float],
    is_best: bool,
    selection_split: str,
    selection_metric: str,
    selection_score: float,
    best_selection_score: float,
    epochs_without_improvement: int,
    patience: int,
) -> list[str]:
    """Format one epoch as stable, human-readable metric groups."""

    lines = [
        f"EPOCH {epoch:03d}/{num_epochs:03d} | "
        f"time={epoch_time_seconds:.2f}s | learning_rate={learning_rate:.6f}"
    ]
    lines.extend(format_split_metrics("TRAIN", train_metrics))
    if dev_metrics:
        lines.extend(format_split_metrics("DEV", dev_metrics))
    selection_name = f"{selection_split.lower()}_{selection_metric}"
    lines.append(
        f"STATUS | improved={str(is_best).lower()} | "
        f"{selection_name}={selection_score:.{METRIC_PRECISION}f} | "
        f"best_{selection_name}={best_selection_score:.{METRIC_PRECISION}f} | "
        f"patience={epochs_without_improvement}/{patience}"
    )
    return lines


def format_split_metrics(split: str, metrics: dict[str, float]) -> list[str]:
    """Group one split's metrics without inventing unavailable values."""

    split_name = split.upper()
    lines: list[str] = []
    _append_group(
        lines,
        split_name,
        "MAIN",
        metrics,
        (
            ("total_loss", "total_loss"),
            ("classification_loss", "classification_loss"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("weighted_f1", "weighted_f1"),
        ),
    )
    _append_group(
        lines,
        split_name,
        "AUX",
        metrics,
        (
            ("relation_contrastive_loss", "relation"),
            ("mode_auxiliary_loss", "mode"),
            ("invariance_loss", "invariance"),
            ("load_balancing_loss", "balancing"),
            ("counterfactual_routing_loss", "counterfactual_routing"),
            (
                "weighted_counterfactual_routing_loss",
                "weighted_counterfactual_routing",
            ),
        ),
    )
    _append_group(
        lines,
        split_name,
        "MODE",
        metrics,
        tuple((f"accuracy_{mode}", f"accuracy_{mode}") for mode in MODE_ABBREVIATIONS),
    )
    _append_group(
        lines,
        split_name,
        "FUSION",
        metrics,
        tuple(
            (f"attention_{mode}_mean", f"attention_{mode}")
            for mode in MODE_ABBREVIATIONS
        )
        + tuple(
            (f"reliability_{mode}_mean", f"reliability_{mode}")
            for mode in MODE_ABBREVIATIONS
        ),
    )
    _append_group(
        lines,
        split_name,
        "ROUTING",
        metrics,
        tuple(
            (f"router_entropy_{mode}", f"entropy_{mode}")
            for mode in MODE_ABBREVIATIONS
        )
        + (
            ("counterfactual_utility_mean", "utility_mean"),
            ("counterfactual_utility_std", "utility_std"),
            ("router_utility_top1_agreement", "utility_top1_agreement"),
        ),
    )
    expert_usage = _format_expert_usage(metrics)
    if expert_usage:
        lines.append(f"{split_name} EXPERTS | {expert_usage}")
    return lines


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append and flush one finite JSON record without replacing prior runs."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=True, allow_nan=False)
    with history_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(serialized + "\n")
        file.flush()


def _append_group(
    lines: list[str],
    split: str,
    group: str,
    metrics: dict[str, float],
    fields: tuple[tuple[str, str], ...],
) -> None:
    values = [
        f"{display_name}={metrics[key]:.{METRIC_PRECISION}f}"
        for key, display_name in fields
        if key in metrics
    ]
    if values:
        lines.append(f"{split} {group} | " + " | ".join(values))


def _format_expert_usage(metrics: dict[str, float]) -> str:
    groups: list[str] = []
    for mode in MODE_ABBREVIATIONS:
        prefix = f"expert_usage_{mode}_expert_"
        matching_keys = [key for key in metrics if key.startswith(prefix)]
        matching_keys.sort(key=lambda key: int(key.removeprefix(prefix)))
        if not matching_keys:
            continue
        values = ",".join(
            f"{key.removeprefix(prefix)}:{metrics[key]:.{METRIC_PRECISION}f}"
            for key in matching_keys
        )
        groups.append(f"{mode}=[{values}]")
    return " | ".join(groups)


__all__ = [
    "METRIC_PRECISION",
    "append_jsonl",
    "create_run_id",
    "format_epoch_summary",
    "format_split_metrics",
]
