from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.schemas import INPUT_MODE_NAMES
from ..models.common.contrastive import NLIRelationContrastiveLoss
from ..models.common.invariance import CrossModalNLIInvarianceLoss


MODE_ABBREVIATIONS = ("TT", "TS", "ST", "SS")


def summarize_mode_values(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    """Summarize a complete [N, M] reliability or attention tensor by mode."""

    if values.ndim != 2:
        raise ValueError(f"values must have shape [N, M], got {tuple(values.shape)}.")
    if values.shape[0] == 0:
        raise ValueError("values must contain at least one sample.")
    if not torch.isfinite(values).all():
        raise ValueError("values must contain only finite values.")

    names = _mode_names(values.shape[1])
    means = values.float().mean(dim=0)
    standard_deviations = values.float().std(dim=0, unbiased=False)
    medians = values.float().median(dim=0).values
    summary: dict[str, float] = {}
    for index, name in enumerate(names):
        summary[f"{prefix}_{name}_mean"] = float(means[index])
        summary[f"{prefix}_{name}_std"] = float(standard_deviations[index])
        summary[f"{prefix}_{name}_median"] = float(medians[index])
    return summary


class EpochMetrics:
    """Aggregate loss, prediction, reliability, attention, and routing metrics."""

    def __init__(
        self,
        relation_criterion: NLIRelationContrastiveLoss,
        invariance_criterion: CrossModalNLIInvarianceLoss,
        *,
        aux_loss_coef: float,
        relation_loss_coef: float,
        use_relation_loss: bool,
        mode_loss_coef: float = 0.0,
        use_mode_loss: bool = False,
        invariance_loss_coef: float = 0.0,
        use_invariance_loss: bool = False,
        reliability_weighted_invariance: bool = False,
        counterfactual_routing_loss_coef: float = 0.0,
        use_counterfactual_routing_loss: bool = False,
        num_labels: int = 3,
        use_load_balancing_loss: bool | None = None,
        report_disabled_losses: bool = True,
    ) -> None:
        self.relation_criterion = relation_criterion
        self.invariance_criterion = invariance_criterion
        self.aux_loss_coef = aux_loss_coef
        self.relation_loss_coef = relation_loss_coef
        self.mode_loss_coef = mode_loss_coef
        self.invariance_loss_coef = invariance_loss_coef
        self.counterfactual_routing_loss_coef = counterfactual_routing_loss_coef
        self.use_relation_loss = use_relation_loss and relation_loss_coef > 0
        self.use_mode_loss = use_mode_loss and mode_loss_coef > 0
        self.use_invariance_loss = use_invariance_loss and invariance_loss_coef > 0
        self.use_counterfactual_routing_loss = (
            use_counterfactual_routing_loss
            and counterfactual_routing_loss_coef > 0
        )
        self.reliability_weighted_invariance = reliability_weighted_invariance
        self.num_labels = num_labels
        self.use_load_balancing_loss = (
            aux_loss_coef > 0
            if use_load_balancing_loss is None
            else use_load_balancing_loss
        )
        self.report_disabled_losses = report_disabled_losses

        self.num_samples = 0
        self.classification_sum = 0.0
        self.balancing_sum = 0.0
        self.saw_load_balancing_output = False
        self.mode_loss_sum = 0.0
        self.invariance_numerator = 0.0
        self.invariance_denominator = 0.0
        self.counterfactual_routing_loss_sum = 0.0
        self.counterfactual_utility_sum = 0.0
        self.counterfactual_utility_squared_sum = 0.0
        self.counterfactual_utility_count = 0
        self.router_utility_agreement_count = 0
        self.router_utility_prediction_count = 0
        self.confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
        self.mode_correct: torch.Tensor | None = None
        self.mode_prediction_count = 0
        self.reliability_batches: list[torch.Tensor] = []
        self.attention_batches: list[torch.Tensor] = []
        self.routing_counts: torch.Tensor | None = None
        self.router_entropy_sum: torch.Tensor | None = None
        self.relation_stats = {
            "entailment_sum": 0.0,
            "entailment_count": 0.0,
            "contradiction_sum": 0.0,
            "contradiction_count": 0.0,
        }

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        losses: dict[str, torch.Tensor],
    ) -> None:
        count = labels.numel()
        self.num_samples += count
        self.classification_sum += float(losses["classification_loss"]) * count
        if "load_balancing_loss" in losses:
            self.saw_load_balancing_output = True
        if self.use_load_balancing_loss:
            # Balancing is batch-dependent; report its sample-weighted mean.
            self.balancing_sum += float(losses["load_balancing_loss"]) * count
        if self.use_mode_loss:
            self.mode_loss_sum += float(losses["mode_auxiliary_loss"]) * count
        if self.use_counterfactual_routing_loss:
            self.counterfactual_routing_loss_sum += (
                float(losses["counterfactual_routing_loss"]) * count
            )
            self._update_counterfactual_utility(outputs)
        self._update_predictions(outputs, labels)
        self._update_mode_outputs(outputs, labels)
        self._update_routing(outputs)

        if self.use_relation_loss:
            stats = self.relation_criterion.loss_statistics(
                outputs["aligned_premise_embeddings"],
                outputs["aligned_hypothesis_embeddings"],
                labels,
            )
            for key, value in stats.items():
                self.relation_stats[key] += float(value)

        if self.use_invariance_loss:
            stats = self.invariance_criterion.loss_statistics(
                outputs["mode_probs"],
                outputs.get("reliability"),
                reliability_weighted=self.reliability_weighted_invariance,
            )
            self.invariance_numerator += float(stats["numerator"])
            self.invariance_denominator += float(stats["denominator"])

    def compute(self) -> dict[str, float]:
        if self.num_samples == 0:
            raise ValueError("Cannot report metrics for an empty DataLoader.")

        classification = self.classification_sum / self.num_samples
        mode_loss = self.mode_loss_sum / self.num_samples if self.use_mode_loss else 0.0
        relation = self._relation_loss()
        invariance = (
            self.invariance_numerator / max(self.invariance_denominator, 1e-8)
            if self.use_invariance_loss
            else 0.0
        )
        counterfactual_routing = (
            self.counterfactual_routing_loss_sum / self.num_samples
            if self.use_counterfactual_routing_loss
            else 0.0
        )
        result: dict[str, float] = {
            "classification_loss": classification,
        }
        weighted_loss_keys: list[str] = []
        if self.use_load_balancing_loss or (
            self.report_disabled_losses and self.saw_load_balancing_output
        ):
            balancing = self.balancing_sum / self.num_samples
            result["load_balancing_loss"] = balancing
            result["weighted_load_balancing_loss"] = self.aux_loss_coef * balancing
            weighted_loss_keys.append("weighted_load_balancing_loss")
        if self.use_relation_loss or self.report_disabled_losses:
            result["relation_contrastive_loss"] = relation
            result["weighted_relation_contrastive_loss"] = self.relation_loss_coef * relation
            if self.use_relation_loss:
                weighted_loss_keys.append("weighted_relation_contrastive_loss")
        if self.use_mode_loss or self.report_disabled_losses:
            result["mode_auxiliary_loss"] = mode_loss
            result["weighted_mode_auxiliary_loss"] = self.mode_loss_coef * mode_loss
            if self.use_mode_loss:
                weighted_loss_keys.append("weighted_mode_auxiliary_loss")
        if self.use_invariance_loss or self.report_disabled_losses:
            result["invariance_loss"] = invariance
            result["weighted_invariance_loss"] = self.invariance_loss_coef * invariance
            if self.use_invariance_loss:
                weighted_loss_keys.append("weighted_invariance_loss")
        if self.use_counterfactual_routing_loss:
            result["counterfactual_routing_loss"] = counterfactual_routing
            result["weighted_counterfactual_routing_loss"] = (
                self.counterfactual_routing_loss_coef
                * counterfactual_routing
            )
            weighted_loss_keys.append("weighted_counterfactual_routing_loss")
        result["total_loss"] = result["classification_loss"] + sum(
            result[key] for key in weighted_loss_keys
        )
        result.update(_classification_metrics(self.confusion))
        result.update(self._mode_metrics())
        result.update(self._routing_metrics())
        result.update(self._counterfactual_utility_metrics())
        if self.reliability_batches:
            result.update(
                summarize_mode_values(
                    torch.cat(self.reliability_batches, dim=0),
                    prefix="reliability",
                )
            )
        if self.attention_batches:
            attention = torch.cat(self.attention_batches, dim=0)
            for index, name in enumerate(_mode_names(attention.shape[1])):
                result[f"attention_{name}_mean"] = float(attention[:, index].mean())
        return result

    def _update_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        predictions = outputs["logits"].argmax(dim=-1)
        encoded = labels * self.num_labels + predictions
        counts = torch.bincount(
            encoded.detach().cpu(),
            minlength=self.num_labels * self.num_labels,
        )
        self.confusion += counts.reshape(self.num_labels, self.num_labels)

    def _update_mode_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        if "mode_logits" in outputs:
            predictions = outputs["mode_logits"].argmax(dim=-1)
            correct = predictions.eq(labels.unsqueeze(1)).sum(dim=0).detach().cpu()
            if self.mode_correct is None:
                self.mode_correct = torch.zeros_like(correct)
            self.mode_correct += correct
            self.mode_prediction_count += labels.numel()
        if "reliability" in outputs:
            self.reliability_batches.append(outputs["reliability"].detach().cpu())
        if "mode_attention_weights" in outputs:
            self.attention_batches.append(
                outputs["mode_attention_weights"].detach().cpu()
            )

    def _update_routing(self, outputs: dict[str, torch.Tensor]) -> None:
        required_keys = {"router_probs", "topk_indices", "topk_weights"}
        present_keys = required_keys & outputs.keys()
        if not present_keys:
            return
        missing_keys = required_keys - outputs.keys()
        if missing_keys:
            raise ValueError(
                "Incomplete routing output; missing "
                f"{', '.join(sorted(missing_keys))}."
            )
        topk_indices = outputs["topk_indices"].detach().cpu()
        router_probabilities = outputs["router_probs"].detach().cpu()
        num_experts = router_probabilities.shape[-1]
        counts = F.one_hot(topk_indices, num_classes=num_experts).sum(dim=(0, 2))
        entropy = -(
            router_probabilities * router_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1).sum(dim=0)
        if self.routing_counts is None:
            self.routing_counts = torch.zeros_like(counts)
            self.router_entropy_sum = torch.zeros_like(entropy)
        self.routing_counts += counts
        self.router_entropy_sum += entropy

    def _update_counterfactual_utility(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> None:
        required_keys = {"router_logits", "counterfactual_utility_targets"}
        missing_keys = required_keys - outputs.keys()
        if missing_keys:
            raise ValueError(
                "Counterfactual utility metrics are missing: "
                f"{', '.join(sorted(missing_keys))}."
            )
        predicted = outputs["router_logits"].detach().cpu()
        targets = outputs["counterfactual_utility_targets"].detach().cpu()
        if predicted.shape != targets.shape:
            raise ValueError(
                "Counterfactual metric tensors must have the same shape, got "
                f"{tuple(predicted.shape)} and {tuple(targets.shape)}."
            )
        targets = targets.double()
        self.counterfactual_utility_sum += float(targets.sum())
        self.counterfactual_utility_squared_sum += float(targets.square().sum())
        self.counterfactual_utility_count += targets.numel()
        self.router_utility_agreement_count += int(
            predicted.argmax(dim=-1).eq(targets.argmax(dim=-1)).sum()
        )
        self.router_utility_prediction_count += targets.shape[0] * targets.shape[1]

    def _relation_loss(self) -> float:
        if not self.use_relation_loss:
            return 0.0
        return sum(
            self.relation_stats[f"{name}_sum"]
            / max(self.relation_stats[f"{name}_count"], 1)
            for name in ("entailment", "contradiction")
        )

    def _mode_metrics(self) -> dict[str, float]:
        if self.mode_correct is None or self.mode_prediction_count == 0:
            return {}
        return {
            f"accuracy_{name}": float(self.mode_correct[index])
            / self.mode_prediction_count
            for index, name in enumerate(_mode_names(self.mode_correct.numel()))
        }

    def _routing_metrics(self) -> dict[str, float]:
        if self.routing_counts is None or self.router_entropy_sum is None:
            return {}
        usage = self.routing_counts.float()
        usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(1)
        result: dict[str, float] = {}
        for mode_index, mode_name in enumerate(_mode_names(usage.shape[0])):
            result[f"router_entropy_{mode_name}"] = float(
                self.router_entropy_sum[mode_index] / self.num_samples
            )
            for expert_index in range(usage.shape[1]):
                result[f"expert_usage_{mode_name}_expert_{expert_index}"] = float(
                    usage[mode_index, expert_index]
                )
        return result

    def _counterfactual_utility_metrics(self) -> dict[str, float]:
        if not self.use_counterfactual_routing_loss:
            return {}
        count = max(self.counterfactual_utility_count, 1)
        mean = self.counterfactual_utility_sum / count
        second_moment = self.counterfactual_utility_squared_sum / count
        variance = max(second_moment - mean * mean, 0.0)
        return {
            "counterfactual_utility_mean": mean,
            "counterfactual_utility_std": variance**0.5,
            "router_utility_top1_agreement": (
                self.router_utility_agreement_count
                / max(self.router_utility_prediction_count, 1)
            ),
        }


def _classification_metrics(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.double()
    true_positives = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    denominator = support + predicted
    f1 = torch.where(
        denominator > 0,
        2.0 * true_positives / denominator,
        torch.zeros_like(denominator),
    )
    total = support.sum().clamp_min(1)
    return {
        "accuracy": float(true_positives.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
    }


def _mode_names(num_modes: int) -> tuple[str, ...]:
    if num_modes == len(INPUT_MODE_NAMES):
        return MODE_ABBREVIATIONS
    return tuple(f"mode_{index}" for index in range(num_modes))
