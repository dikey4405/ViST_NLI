from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data.config import DEFAULT_LABEL_MAPPING
from ..models.common.contrastive import NLIRelationContrastiveLoss
from ..models.common.invariance import CrossModalNLIInvarianceLoss
from ..models.common.losses import compute_mode_auxiliary_loss
from ..models.common.routing_utils import compute_routing_statistics
from .feature_dataset import FeatureBatchCollator, FeatureTensorDataset
from .logging_utils import (
    append_jsonl,
    create_run_id,
    format_epoch_summary,
    format_split_metrics,
)
from .model_factory import build_model
from .metrics import EpochMetrics, summarize_mode_values
from .utils import Timer, count_parameters, load_yaml_config, resolve_path, set_random_seed


SELECTION_METRIC = "macro_f1"
SELECTION_MODE = "max"


def differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_training_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    aux_loss_coef: float,
    relation_contrastive_loss_coef: float,
    relation_contrastive_criterion: NLIRelationContrastiveLoss,
    use_relation_contrastive_loss: bool,
    mode_auxiliary_loss_coef: float = 0.0,
    use_mode_auxiliary_loss: bool = False,
    invariance_loss_coef: float = 0.0,
    use_invariance_loss: bool = False,
    invariance_criterion: CrossModalNLIInvarianceLoss | None = None,
    reliability_weighted_invariance: bool = False,
    counterfactual_routing_loss_coef: float = 0.0,
    use_counterfactual_routing_loss: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute final, routing, relation, mode, and invariance objectives."""

    if aux_loss_coef < 0:
        raise ValueError(f"aux_loss_coef must be non-negative, got {aux_loss_coef}.")
    if relation_contrastive_loss_coef < 0:
        raise ValueError(
            "relation_contrastive_loss_coef must be non-negative, "
            f"got {relation_contrastive_loss_coef}."
        )
    if mode_auxiliary_loss_coef < 0:
        raise ValueError(
            f"mode_auxiliary_loss_coef must be non-negative, got {mode_auxiliary_loss_coef}."
        )
    if invariance_loss_coef < 0:
        raise ValueError(
            f"invariance_loss_coef must be non-negative, got {invariance_loss_coef}."
        )
    if counterfactual_routing_loss_coef < 0:
        raise ValueError(
            "counterfactual_routing_loss_coef must be non-negative, "
            f"got {counterfactual_routing_loss_coef}."
        )

    classification_loss = F.cross_entropy(outputs["logits"], labels)
    if "load_balancing_loss" in outputs:
        load_balancing_loss = outputs["load_balancing_loss"]
    elif aux_loss_coef > 0:
        raise ValueError(
            "aux_loss_coef is positive, but the model has no load_balancing_loss output."
        )
    else:
        load_balancing_loss = differentiable_zero(outputs["logits"])
    should_use_relation_loss = use_relation_contrastive_loss and relation_contrastive_loss_coef > 0
    if should_use_relation_loss:
        required_keys = {"aligned_premise_embeddings", "aligned_hypothesis_embeddings"}
        missing_keys = required_keys - outputs.keys()
        if missing_keys:
            raise ValueError(
                "Relation contrastive loss is enabled, but model output is missing: "
                f"{', '.join(sorted(missing_keys))}."
            )
        relation_contrastive_loss = relation_contrastive_criterion(
            outputs["aligned_premise_embeddings"],
            outputs["aligned_hypothesis_embeddings"],
            labels,
        )
    else:
        relation_contrastive_loss = differentiable_zero(outputs["logits"])

    should_use_mode_loss = use_mode_auxiliary_loss and mode_auxiliary_loss_coef > 0
    if should_use_mode_loss:
        if "mode_logits" not in outputs:
            raise ValueError(
                "Mode auxiliary loss is enabled, but model output is missing mode_logits."
            )
        mode_auxiliary_loss = compute_mode_auxiliary_loss(outputs["mode_logits"], labels)
    else:
        mode_auxiliary_loss = differentiable_zero(outputs["logits"])

    should_use_invariance = use_invariance_loss and invariance_loss_coef > 0
    if should_use_invariance:
        if invariance_criterion is None:
            raise ValueError("Invariance loss is enabled, but no criterion was provided.")
        required_keys = {"mode_probs"}
        if reliability_weighted_invariance:
            required_keys.add("reliability")
        missing_keys = required_keys - outputs.keys()
        if missing_keys:
            raise ValueError(
                "Invariance loss is enabled, but model output is missing: "
                f"{', '.join(sorted(missing_keys))}."
            )
        invariance_loss = invariance_criterion(
            outputs["mode_probs"],
            outputs.get("reliability"),
            reliability_weighted=reliability_weighted_invariance,
        )
    else:
        invariance_loss = differentiable_zero(outputs["logits"])

    should_use_counterfactual_routing = (
        use_counterfactual_routing_loss
        and counterfactual_routing_loss_coef > 0
    )
    if should_use_counterfactual_routing:
        required_keys = {"router_logits", "counterfactual_utility_targets"}
        missing_keys = required_keys - outputs.keys()
        if missing_keys:
            raise ValueError(
                "Counterfactual routing loss is enabled, but model output is missing: "
                f"{', '.join(sorted(missing_keys))}."
            )
        predicted_utility = outputs["router_logits"]
        target_utility = outputs["counterfactual_utility_targets"]
        if predicted_utility.shape != target_utility.shape:
            raise ValueError(
                "router_logits and counterfactual_utility_targets must have the "
                f"same shape, got {tuple(predicted_utility.shape)} and "
                f"{tuple(target_utility.shape)}."
            )
        counterfactual_routing_loss = F.smooth_l1_loss(
            predicted_utility,
            target_utility.detach(),
        )
    else:
        counterfactual_routing_loss = differentiable_zero(outputs["logits"])

    weighted_load_balancing_loss = aux_loss_coef * load_balancing_loss
    weighted_relation_contrastive_loss = (
        relation_contrastive_loss_coef * relation_contrastive_loss
    )
    weighted_mode_auxiliary_loss = mode_auxiliary_loss_coef * mode_auxiliary_loss
    weighted_invariance_loss = invariance_loss_coef * invariance_loss
    weighted_counterfactual_routing_loss = (
        counterfactual_routing_loss_coef * counterfactual_routing_loss
    )
    total_loss = (
        classification_loss
        + weighted_load_balancing_loss
        + weighted_relation_contrastive_loss
        + weighted_mode_auxiliary_loss
        + weighted_invariance_loss
        + weighted_counterfactual_routing_loss
    )
    return {
        "classification_loss": classification_loss,
        "load_balancing_loss": load_balancing_loss,
        "weighted_load_balancing_loss": weighted_load_balancing_loss,
        "relation_contrastive_loss": relation_contrastive_loss,
        "weighted_relation_contrastive_loss": weighted_relation_contrastive_loss,
        "mode_auxiliary_loss": mode_auxiliary_loss,
        "weighted_mode_auxiliary_loss": weighted_mode_auxiliary_loss,
        "invariance_loss": invariance_loss,
        "weighted_invariance_loss": weighted_invariance_loss,
        "counterfactual_routing_loss": counterfactual_routing_loss,
        "weighted_counterfactual_routing_loss": (
            weighted_counterfactual_routing_loss
        ),
        "total_loss": total_loss,
    }


class Trainer:
    """Shared trainer that depends only on the common model output interface."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader | None,
        config: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.routing_strategy = getattr(model, "routing_strategy", None)
        self.counterfactual_routing_loss_coef = float(
            config.get("counterfactual_routing_loss_coef", 0.0)
        )
        if self.counterfactual_routing_loss_coef < 0:
            raise ValueError(
                "counterfactual_routing_loss_coef must be non-negative, "
                f"got {self.counterfactual_routing_loss_coef}."
            )
        configured_strategy = self._routing_strategy_from_config(config)
        if (
            self.routing_strategy is not None
            and configured_strategy != self.routing_strategy
        ):
            raise ValueError(
                f"Config routing strategy {configured_strategy!r} does not match "
                f"model routing strategy {self.routing_strategy!r}."
            )
        if (
            self.counterfactual_routing_loss_coef > 0
            and self.routing_strategy != "counterfactual_utility"
        ):
            raise ValueError(
                "counterfactual_routing_loss_coef is positive, but the model does "
                "not use counterfactual_utility routing."
            )
        self.use_counterfactual_routing_loss = (
            self.routing_strategy == "counterfactual_utility"
            and self.counterfactual_routing_loss_coef > 0
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = create_run_id()
        self.metrics_history_path = self.output_dir / "metrics_history.jsonl"
        self.logger = self._build_logger()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.relation_contrastive_criterion = NLIRelationContrastiveLoss(
            margin=float(config.get("relation_contrastive_margin", 0.5)),
            entailment_label_id=DEFAULT_LABEL_MAPPING["entailment"],
            contradiction_label_id=DEFAULT_LABEL_MAPPING["contradiction"],
        )
        self.invariance_criterion = CrossModalNLIInvarianceLoss()

    def train(self) -> None:
        best_selection_score = float("-inf")
        epochs_without_improvement = 0
        num_epochs = int(self.config["num_epochs"])
        patience = int(self.config["patience"])
        if num_epochs < 1 or patience < 1:
            raise ValueError("num_epochs and patience must both be positive.")
        parameter_counts = count_parameters(self.model)
        self.logger.info(
            "RUN START | run_id=%s | model=%s | feature_source=%s | device=%s",
            self.run_id,
            self.config.get("model_name", type(self.model).__name__),
            self.config.get("feature_source", "legacy"),
            self.device,
        )
        self.logger.info(
            "CONFIG | epochs=%d | batch_size=%d | learning_rate=%.6f | "
            "patience=%d | selection_metric=%s | selection_mode=%s",
            num_epochs,
            int(self.config["batch_size"]),
            float(self.config["learning_rate"]),
            patience,
            SELECTION_METRIC,
            SELECTION_MODE,
        )
        self.logger.info(
            "PARAMETERS | total=%d | trainable=%d",
            parameter_counts["total_parameters"],
            parameter_counts["trainable_parameters"],
        )

        with Timer() as total_timer:
            for epoch in range(1, num_epochs + 1):
                with Timer() as epoch_timer:
                    train_metrics = self._run_epoch(training=True)
                    dev_metrics = (
                        self._run_epoch(training=False)
                        if self.dev_loader is not None
                        else {}
                    )

                selection_metrics = dev_metrics if dev_metrics else train_metrics
                selection_split = "dev" if dev_metrics else "train"
                selection_score = selection_metrics[SELECTION_METRIC]
                is_best = selection_score > best_selection_score
                if is_best:
                    best_selection_score = selection_score
                    epochs_without_improvement = 0
                    self._save_checkpoint(
                        "best_model.pt",
                        epoch,
                        best_selection_score,
                    )
                else:
                    epochs_without_improvement += 1

                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                for line in format_epoch_summary(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    epoch_time_seconds=epoch_timer.elapsed,
                    learning_rate=learning_rate,
                    train_metrics=train_metrics,
                    dev_metrics=dev_metrics,
                    is_best=is_best,
                    selection_split=selection_split,
                    selection_metric=SELECTION_METRIC,
                    selection_score=selection_score,
                    best_selection_score=best_selection_score,
                    epochs_without_improvement=epochs_without_improvement,
                    patience=patience,
                ):
                    self.logger.info("%s", line)
                append_jsonl(
                    self.metrics_history_path,
                    {
                        "run_id": self.run_id,
                        "epoch": epoch,
                        "num_epochs": num_epochs,
                        "epoch_time_seconds": epoch_timer.elapsed,
                        "learning_rate": learning_rate,
                        "is_best": is_best,
                        "epochs_without_improvement": epochs_without_improvement,
                        "selection_metric": SELECTION_METRIC,
                        "selection_score": selection_score,
                        "best_selection_score": best_selection_score,
                        "train": train_metrics,
                        "dev": dev_metrics,
                    },
                )

                if torch.cuda.is_available():
                    self.logger.info(
                        "GPU | peak_memory_mb=%.2f",
                        torch.cuda.max_memory_allocated() / (1024**2),
                    )
                if epochs_without_improvement >= patience:
                    self.logger.info("EARLY STOPPING | epoch=%d", epoch)
                    break

        self.logger.info(
            "RUN END | run_id=%s | total_time=%.2fs | "
            "selection_metric=%s | best_selection_score=%.6f",
            self.run_id,
            total_timer.elapsed,
            SELECTION_METRIC,
            best_selection_score,
        )

    def _run_epoch(
        self, *, training: bool, loader: DataLoader | None = None,
    ) -> dict[str, float]:
        if loader is None:
            loader = self.train_loader if training else self.dev_loader
        if loader is None:
            return {}

        self.model.train(training)
        metrics = EpochMetrics(
            self.relation_contrastive_criterion,
            self.invariance_criterion,
            aux_loss_coef=float(self.config.get("aux_loss_coef", 0.0)),
            relation_loss_coef=float(
                self.config.get("relation_contrastive_loss_coef", 0.0)
            ),
            use_relation_loss=bool(
                self.config.get("use_relation_contrastive_loss", False)
            ),
            mode_loss_coef=float(self.config.get("mode_auxiliary_loss_coef", 0.0)),
            use_mode_loss=bool(self.config.get("use_mode_auxiliary_loss", False)),
            invariance_loss_coef=float(self.config.get("invariance_loss_coef", 0.0)),
            use_invariance_loss=bool(self.config.get("use_invariance_loss", False)),
            reliability_weighted_invariance=bool(
                self.config.get("reliability_weighted_invariance", False)
            ),
            counterfactual_routing_loss_coef=(
                self.counterfactual_routing_loss_coef
            ),
            use_counterfactual_routing_loss=(
                self.use_counterfactual_routing_loss
            ),
            num_labels=int(self.config.get("num_labels", 3)),
            report_disabled_losses=hasattr(self.model, "router"),
        )

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.set_grad_enabled(training):
                outputs = self.model(features)
                if self.use_counterfactual_routing_loss:
                    target_builder = getattr(
                        self.model,
                        "build_counterfactual_utility_targets",
                        None,
                    )
                    if target_builder is None:
                        raise TypeError(
                            "Counterfactual utility routing requires a model target builder."
                        )
                    outputs["counterfactual_utility_targets"] = target_builder(
                        features,
                        labels,
                    )
                losses = compute_training_losses(
                    outputs,
                    labels,
                    aux_loss_coef=float(self.config.get("aux_loss_coef", 0.0)),
                    relation_contrastive_loss_coef=float(
                        self.config.get("relation_contrastive_loss_coef", 0.0)
                    ),
                    relation_contrastive_criterion=self.relation_contrastive_criterion,
                    use_relation_contrastive_loss=bool(
                        self.config.get("use_relation_contrastive_loss", False)
                    ),
                    mode_auxiliary_loss_coef=float(
                        self.config.get("mode_auxiliary_loss_coef", 0.0)
                    ),
                    use_mode_auxiliary_loss=bool(
                        self.config.get("use_mode_auxiliary_loss", False)
                    ),
                    invariance_loss_coef=float(
                        self.config.get("invariance_loss_coef", 0.0)
                    ),
                    use_invariance_loss=bool(
                        self.config.get("use_invariance_loss", False)
                    ),
                    invariance_criterion=self.invariance_criterion,
                    reliability_weighted_invariance=bool(
                        self.config.get("reliability_weighted_invariance", False)
                    ),
                    counterfactual_routing_loss_coef=(
                        self.counterfactual_routing_loss_coef
                    ),
                    use_counterfactual_routing_loss=(
                        self.use_counterfactual_routing_loss
                    ),
                )
                if not all(torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError("Non-finite training/evaluation loss detected.")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total_loss"].backward()
                    self.optimizer.step()

            metrics.update(outputs, labels, losses)

        return metrics.compute()

    def evaluate_test(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate held-out data once using the checkpoint selected only on dev."""

        checkpoint_path = self.output_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._validate_checkpoint_routing_strategy(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        selection_metric, selection_mode = self._checkpoint_selection_metadata(
            checkpoint
        )
        report = {
            "feature_source": self.config["feature_source"],
            "model_name": self.config["model_name"],
            "checkpoint": str(checkpoint_path.resolve()),
            "best_epoch": checkpoint["epoch"],
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_score": checkpoint["score"],
            "num_samples": len(loader.dataset),
            "metrics": self._run_epoch(training=False, loader=loader),
        }
        with (self.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
        self.logger.info(
            "TEST | run_id=%s | best_epoch=%d | selection_metric=%s | "
            "best_selection_score=%.6f | samples=%d",
            self.run_id,
            report["best_epoch"],
            report["selection_metric"],
            report["best_selection_score"],
            report["num_samples"],
        )
        for line in format_split_metrics("TEST", report["metrics"]):
            self.logger.info("%s", line)
        self.logger.info("TEST OUTPUT | path=%s", self.output_dir / "test_metrics.json")
        return report

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def collect_routing_statistics(self, loader: DataLoader) -> list[dict[str, Any]]:
        self.model.eval()
        stats: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["features"].to(self.device))
                required_keys = {"router_probs", "topk_indices", "topk_weights"}
                missing_keys = required_keys - outputs.keys()
                if missing_keys:
                    raise ValueError(
                        "The model does not provide routing outputs: "
                        f"{', '.join(sorted(missing_keys))}."
                    )
                stat = compute_routing_statistics(
                    outputs["router_probs"],
                    outputs["topk_indices"],
                    outputs["topk_weights"],
                )
                stat["load_balancing_loss"] = outputs["load_balancing_loss"].detach().cpu()
                stats.append(stat)
        return stats

    def collect_reliability(self, loader: DataLoader) -> torch.Tensor:
        """Collect per-sample reliability in fixed TT/TS/ST/SS order."""

        self.model.eval()
        batches: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch["features"].to(self.device))
                if "reliability" not in outputs:
                    raise ValueError(
                        "The model has no reliability output; enable a mode-evidence feature."
                    )
                batches.append(outputs["reliability"].detach().cpu())
        if not batches:
            raise ValueError("Cannot collect reliability from an empty DataLoader.")
        return torch.cat(batches, dim=0)

    def summarize_reliability(self, loader: DataLoader) -> dict[str, float]:
        """Return reliability mean, standard deviation, and median by mode."""

        return summarize_mode_values(
            self.collect_reliability(loader),
            prefix="reliability",
        )

    def _save_checkpoint(self, file_name: str, epoch: int, score: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "selection_metric": SELECTION_METRIC,
                "selection_mode": SELECTION_MODE,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            self.output_dir / file_name,
        )

    @staticmethod
    def _checkpoint_selection_metadata(
        checkpoint: dict[str, Any],
    ) -> tuple[str, str]:
        """Return selection metadata, recognizing pre-macro-F1 checkpoints."""

        selection_metric = checkpoint.get("selection_metric")
        selection_mode = checkpoint.get("selection_mode")
        if selection_metric is None and selection_mode is None:
            return "total_loss", "min"
        if not isinstance(selection_metric, str) or not selection_metric:
            raise ValueError("Checkpoint selection_metric must be a non-empty string.")
        if selection_mode not in {"min", "max"}:
            raise ValueError(
                "Checkpoint selection_mode must be either 'min' or 'max'."
            )
        return selection_metric, selection_mode

    @staticmethod
    def _routing_strategy_from_config(config: dict[str, Any]) -> str:
        if "routing_strategy" in config:
            return str(config["routing_strategy"])
        if "use_reliability_routing" in config:
            return (
                "reliability"
                if bool(config["use_reliability_routing"])
                else "topk"
            )
        return "topk"

    def _validate_checkpoint_routing_strategy(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.routing_strategy is None:
            return
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError(
                "Checkpoint is missing config metadata required to validate routing."
            )
        checkpoint_strategy = self._routing_strategy_from_config(checkpoint_config)
        if checkpoint_strategy != self.routing_strategy:
            raise ValueError(
                f"Checkpoint routing strategy {checkpoint_strategy!r} does not match "
                f"model routing strategy {self.routing_strategy!r}."
            )

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"trainer.{self.output_dir.resolve()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger


def build_feature_loader(path: str | Path, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    expected_feature_dim = int(config.get("input_dim", 4096))
    dataset = FeatureTensorDataset(
        path,
        label_mapping=DEFAULT_LABEL_MAPPING,
        expected_feature_dim=expected_feature_dim,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        collate_fn=FeatureBatchCollator(expected_feature_dim=expected_feature_dim),
    )


def resolve_feature_source_paths(
    config: dict[str, Any],
    *,
    feature_source: str | None = None,
) -> tuple[str, dict[str, Path]]:
    """Resolve exactly one configured feature source for the current training run."""

    selected_source = config.get("feature_source") if feature_source is None else feature_source
    feature_sources = config.get("feature_sources")
    if selected_source is None and feature_sources is None:
        required = ("train_path", "dev_path", "test_path")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Legacy feature paths are missing: {missing}.")
        return "legacy", {
            split: resolve_path(config[f"{split}_path"])
            for split in ("train", "dev", "test")
        }
    if not isinstance(selected_source, str) or not selected_source:
        raise ValueError("feature_source must name exactly one entry in feature_sources.")
    if not isinstance(feature_sources, dict):
        raise ValueError("feature_sources must be a mapping of named feature path sets.")
    if selected_source not in feature_sources:
        raise ValueError(
            f"Unknown feature_source '{selected_source}'. "
            f"Available sources: {sorted(feature_sources)}."
        )
    selected = feature_sources[selected_source]
    if not isinstance(selected, dict):
        raise ValueError(f"feature_sources.{selected_source} must be a mapping.")
    missing_splits = [split for split in ("train", "dev", "test") if split not in selected]
    if missing_splits:
        raise ValueError(
            f"feature_sources.{selected_source} is missing splits: {missing_splits}."
        )
    return selected_source, {
        split: resolve_path(selected[split])
        for split in ("train", "dev", "test")
    }


def resolve_training_output_dir(
    output_dir: str | Path,
    feature_source: str,
) -> Path:
    """Resolve a run directory, expanding the configured feature-source token."""

    rendered_path = str(output_dir).replace("{feature_source}", feature_source)
    return resolve_path(rendered_path)


def run_training_from_config(
    config_path: str | Path,
    *,
    feature_source: str | None = None,
) -> None:
    """Train one NLI model from exactly one saved feature source."""

    config = load_yaml_config(config_path)
    set_random_seed(int(config["seed"]))
    selected_source, feature_paths = resolve_feature_source_paths(
        config,
        feature_source=feature_source,
    )
    config["feature_source"] = selected_source
    for split, path in feature_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {selected_source} {split} features: {path}")
    output_dir = resolve_training_output_dir(
        config["output_dir"],
        selected_source,
    )
    config["output_dir"] = str(output_dir)
    model = build_model(config)
    train_loader = build_feature_loader(
        feature_paths["train"],
        config,
        shuffle=True,
    )
    dev_loader = build_feature_loader(
        feature_paths["dev"],
        config,
        shuffle=False,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=config,
        output_dir=output_dir,
    )
    try:
        trainer.train()
        test_loader = build_feature_loader(feature_paths["test"], config, shuffle=False)
        trainer.evaluate_test(test_loader)
    finally:
        trainer.close()
