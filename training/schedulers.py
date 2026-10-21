from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


NO_SCHEDULER = "none"
COSINE_WITH_WARMUP = "cosine_with_warmup"
SUPPORTED_SCHEDULERS = frozenset({NO_SCHEDULER, COSINE_WITH_WARMUP})


def resolve_scheduler_name(config: dict[str, Any]) -> str:
    """Return the normalized scheduler name, defaulting to a fixed LR."""

    name = str(config.get("scheduler_name", NO_SCHEDULER)).strip().lower()
    if name not in SUPPORTED_SCHEDULERS:
        supported = ", ".join(sorted(SUPPORTED_SCHEDULERS))
        raise ValueError(
            f"Unsupported scheduler_name {name!r}. Expected one of: {supported}."
        )
    return name


def build_lr_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
    *,
    steps_per_epoch: int,
) -> LRScheduler | None:
    """Build an optimizer-step scheduler from the flat training config."""

    scheduler_name = resolve_scheduler_name(config)
    if scheduler_name == NO_SCHEDULER:
        return None

    num_epochs = int(config["num_epochs"])
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}.")
    if steps_per_epoch < 1:
        raise ValueError(
            "steps_per_epoch must be positive when a scheduler is enabled, "
            f"got {steps_per_epoch}."
        )

    warmup_ratio = float(config.get("scheduler_warmup_ratio", 0.0))
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "scheduler_warmup_ratio must be in [0, 1), "
            f"got {warmup_ratio}."
        )

    min_learning_rate = float(config.get("scheduler_min_learning_rate", 0.0))
    if not math.isfinite(min_learning_rate) or min_learning_rate < 0.0:
        raise ValueError(
            "scheduler_min_learning_rate must be non-negative, "
            f"got {min_learning_rate}."
        )

    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not base_learning_rates or any(
        not math.isfinite(rate) or rate <= 0.0
        for rate in base_learning_rates
    ):
        raise ValueError("All optimizer learning rates must be finite and positive.")
    if any(min_learning_rate > rate for rate in base_learning_rates):
        raise ValueError(
            "scheduler_min_learning_rate must not exceed any optimizer "
            f"learning rate, got min={min_learning_rate} and "
            f"base={base_learning_rates}."
        )

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_ratio > 0.0 and total_steps > 1:
        warmup_steps = max(1, warmup_steps)
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    lr_lambdas = [
        _cosine_with_warmup_lambda(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_learning_rate / base_learning_rate,
        )
        for base_learning_rate in base_learning_rates
    ]
    return LambdaLR(optimizer, lr_lambda=lr_lambdas)


def _cosine_with_warmup_lambda(
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Create a half-cosine multiplier over the planned optimizer updates."""

    decay_steps = total_steps - warmup_steps

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if decay_steps <= 1:
            progress = 1.0 if warmup_steps > 0 and step >= warmup_steps else 0.0
        else:
            progress = (step - warmup_steps) / float(decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return multiplier


__all__ = [
    "COSINE_WITH_WARMUP",
    "NO_SCHEDULER",
    "build_lr_scheduler",
    "resolve_scheduler_name",
]
