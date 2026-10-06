from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


KLTN_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = KLTN_ROOT.parent


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must contain a YAML mapping: {config_path}")
    return payload


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a training config independently of the current working directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        return resolved

    candidates: list[Path] = []
    if candidate.parts and candidate.parts[0].lower() == KLTN_ROOT.name.lower():
        candidates.append(WORKSPACE_ROOT / candidate)
    else:
        candidates.extend(
            (
                Path.cwd() / candidate,
                KLTN_ROOT / candidate,
                KLTN_ROOT / "source" / "configs" / candidate,
            )
        )
    for config_path in candidates:
        if config_path.exists():
            return config_path.resolve()
    searched = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Training config not found. Checked: {searched}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    """Resolve config values relative to KLTN, with legacy KLTN/... support."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0].lower() == KLTN_ROOT.name.lower():
        return (WORKSPACE_ROOT / candidate).resolve()
    return (KLTN_ROOT / candidate).resolve()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    expert = sum(param.numel() for name, param in model.named_parameters() if "expert" in name)
    alignment = sum(
        param.numel()
        for name, param in model.named_parameters()
        if "alignment_projection" in name
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "expert_parameters": expert,
        "alignment_parameters": alignment,
    }


class Timer:
    """Simple context manager for epoch/training time logging."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
