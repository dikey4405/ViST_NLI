from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...pair_features import PAIR_FEATURE_COMPONENTS, build_pair_feature, split_pair_feature
from .contrastive import PairAlignmentProjection
from .expert import FeedForwardExpert, InputProjection, ModeNLIHead, NLIClassifier
from .losses import compute_load_balancing_loss
from .pooling import ModeAttentionPooling
from .reliability import EntropyReliabilityEstimator
from .router import CounterfactualUtilityRouter, ReliabilityAwareRouter, TopKRouter
from .routing_utils import combine_topk_expert_outputs, reshape_router_tensor, validate_feature_tensor


class RoutedMoENLIModel(nn.Module):
    """Common routed-MoE implementation shared by all NLI model variants."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_modes: int,
        hidden_dim: int,
        num_routed_experts: int,
        routed_top_k: int,
        expert_ffn_dim: int,
        num_labels: int,
        dropout: float,
        use_relation_contrastive_loss: bool,
        alignment_hidden_dim: int,
        num_shared_experts: int = 0,
        use_mode_evidence: bool = False,
        use_reliability_routing: bool = False,
        use_reliability_fusion: bool = False,
        reliability_estimator: str = "entropy",
        detach_reliability_for_routing: bool = True,
        detach_reliability_for_fusion: bool = True,
        mode_embedding_dim: int = 32,
        reliability_embedding_dim: int = 16,
        routing_strategy: str | None = None,
    ) -> None:
        super().__init__()
        if input_dim % PAIR_FEATURE_COMPONENTS != 0:
            raise ValueError(
                f"input_dim must be divisible by {PAIR_FEATURE_COMPONENTS}, got {input_dim}."
            )
        self.input_dim = input_dim
        self.embedding_dim = input_dim // PAIR_FEATURE_COMPONENTS
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.routed_top_k = routed_top_k
        self.use_relation_contrastive_loss = use_relation_contrastive_loss
        self.routing_strategy = self._resolve_routing_strategy(
            routing_strategy,
            use_reliability_routing,
        )
        self.use_reliability_routing = self.routing_strategy == "reliability"
        self.use_reliability_fusion = use_reliability_fusion
        self.detach_reliability_for_routing = detach_reliability_for_routing
        self.detach_reliability_for_fusion = detach_reliability_for_fusion
        self.use_mode_evidence = (
            use_mode_evidence
            or self.use_reliability_routing
            or use_reliability_fusion
        )
        if reliability_estimator != "entropy":
            raise ValueError(
                "Only entropy reliability is supported, "
                f"got '{reliability_estimator}'."
            )

        self.alignment_projection = (
            PairAlignmentProjection(self.embedding_dim, alignment_hidden_dim, dropout)
            if use_relation_contrastive_loss
            else None
        )

        self.input_projection = InputProjection(input_dim, hidden_dim, dropout)
        self.mode_nli_head = (
            ModeNLIHead(hidden_dim, num_labels, dropout)
            if self.use_mode_evidence
            else None
        )
        self.reliability_estimator = (
            EntropyReliabilityEstimator(num_labels)
            if self.use_mode_evidence
            else None
        )
        if self.routing_strategy in {"reliability", "counterfactual_utility"}:
            self.mode_embedding = nn.Embedding(num_modes, mode_embedding_dim)
        else:
            self.mode_embedding = None

        if self.routing_strategy == "reliability":
            self.router = ReliabilityAwareRouter(
                hidden_dim,
                num_routed_experts,
                routed_top_k,
                mode_embedding_dim=mode_embedding_dim,
                reliability_embedding_dim=reliability_embedding_dim,
            )
        elif self.routing_strategy == "counterfactual_utility":
            self.router = CounterfactualUtilityRouter(
                hidden_dim,
                num_routed_experts,
                routed_top_k,
                mode_embedding_dim=mode_embedding_dim,
            )
        else:
            self.router = TopKRouter(hidden_dim, num_routed_experts, routed_top_k)
        self.routed_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_routed_experts)]
        )

        if num_shared_experts < 0:
            raise ValueError(
                f"num_shared_experts must be non-negative, got {num_shared_experts}."
            )
        self.num_shared_experts = num_shared_experts
        self.shared_experts = nn.ModuleList(
            [FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_shared_experts)]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mode_pooling = ModeAttentionPooling(
            hidden_dim,
            use_reliability=use_reliability_fusion,
            reliability_hidden_dim=reliability_embedding_dim,
        )
        self.classifier = NLIClassifier(hidden_dim, num_labels, dropout)

    @staticmethod
    def _resolve_routing_strategy(
        routing_strategy: str | None,
        use_reliability_routing: bool,
    ) -> str:
        if routing_strategy is None:
            return "reliability" if use_reliability_routing else "topk"
        valid_strategies = {"topk", "reliability", "counterfactual_utility"}
        if routing_strategy not in valid_strategies:
            raise ValueError(
                f"routing_strategy must be one of {sorted(valid_strategies)}, "
                f"got {routing_strategy!r}."
            )
        if use_reliability_routing and routing_strategy != "reliability":
            raise ValueError(
                "use_reliability_routing=true conflicts with routing_strategy="
                f"{routing_strategy!r}."
            )
        return routing_strategy

    def build_classification_features(
        self,
        premise_embeddings: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Build classifier input from the original, unaligned embeddings."""

        return build_pair_feature(
            premise_embeddings,
            hypothesis_embeddings,
            expected_embedding_dim=self.embedding_dim,
        )

    def build_relation_embeddings(
        self,
        premise_embeddings: torch.Tensor,
        hypothesis_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Project a pair only for the auxiliary relation objective."""

        if self.alignment_projection is None:
            return None
        return (
            self.alignment_projection(premise_embeddings),
            self.alignment_projection(hypothesis_embeddings),
        )

    def _apply_moe_block(
        self,
        flat: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        routed_flat = combine_topk_expert_outputs(
            flat,
            self.routed_experts,
            routing["topk_indices"],
            routing["topk_weights"],
        )
        shared_flat: torch.Tensor | None = None
        combined = flat + routed_flat
        if self.num_shared_experts > 0:
            shared_flat = sum(
                (expert(flat) for expert in self.shared_experts),
                torch.zeros_like(flat),
            )
            combined = combined + shared_flat
        return self.output_norm(combined), routed_flat, shared_flat

    def _build_mode_condition(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode_embedding is None:
            raise RuntimeError(
                "Mode embeddings are unavailable for the selected routing strategy."
            )
        mode_ids = torch.arange(self.num_modes, device=device)
        mode_ids = mode_ids.unsqueeze(0).expand(batch_size, -1)
        return mode_ids, self.mode_embedding(mode_ids)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_modes = validate_feature_tensor(
            features,
            input_dim=self.input_dim,
            num_modes=self.num_modes,
        )

        premise_embeddings, hypothesis_embeddings = split_pair_feature(
            features,
            expected_embedding_dim=self.embedding_dim,
        )
        classification_features = self.build_classification_features(
            premise_embeddings,
            hypothesis_embeddings,
        )
        projected = self.input_projection(classification_features)
        mode_logits: torch.Tensor | None = None
        mode_probabilities: torch.Tensor | None = None
        reliability: torch.Tensor | None = None
        if self.mode_nli_head is not None and self.reliability_estimator is not None:
            mode_logits = self.mode_nli_head(projected)
            mode_probabilities = torch.softmax(mode_logits, dim=-1)
            reliability = self.reliability_estimator(mode_logits)

        flat = projected.reshape(batch_size * num_modes, self.hidden_dim)
        if self.routing_strategy in {"reliability", "counterfactual_utility"}:
            _, mode_condition = self._build_mode_condition(
                batch_size,
                features.device,
            )
            flat_mode_condition = mode_condition.reshape(batch_size * num_modes, -1)
        else:
            flat_mode_condition = None

        if self.routing_strategy == "reliability":
            if reliability is None:
                raise RuntimeError("Reliability-aware routing requires mode reliability.")
            routing_reliability = (
                reliability.detach()
                if self.detach_reliability_for_routing
                else reliability
            )
            routing = self.router(
                flat,
                flat_mode_condition,
                routing_reliability.reshape(batch_size * num_modes),
            )
        elif self.routing_strategy == "counterfactual_utility":
            routing = self.router(flat, flat_mode_condition)
        else:
            routing = self.router(flat)
        moe_flat, routed_flat, shared_flat = self._apply_moe_block(flat, routing)

        moe_output = moe_flat.reshape(batch_size, num_modes, self.hidden_dim)
        fusion_reliability: torch.Tensor | None = None
        if self.use_reliability_fusion:
            if reliability is None:
                raise RuntimeError("Reliability-aware fusion requires mode reliability.")
            fusion_reliability = (
                reliability.detach()
                if self.detach_reliability_for_fusion
                else reliability
            )
        fused, mode_attention_weights = self.mode_pooling(
            moe_output,
            fusion_reliability,
        )
        output = {
            "logits": self.classifier(fused),
            "fused": fused,
            "pre_moe_output": projected,
            "moe_output": moe_output,
            "mode_attention_weights": mode_attention_weights,
            "router_logits": reshape_router_tensor(routing["router_logits"], batch_size, num_modes),
            "router_probs": reshape_router_tensor(routing["router_probs"], batch_size, num_modes),
            "topk_indices": reshape_router_tensor(routing["topk_indices"], batch_size, num_modes),
            "topk_weights": reshape_router_tensor(routing["topk_weights"], batch_size, num_modes),
            "load_balancing_loss": compute_load_balancing_loss(
                routing["router_probs"],
                routing["topk_indices"],
                self.num_routed_experts,
            ),
        }

        if mode_logits is not None and mode_probabilities is not None and reliability is not None:
            output["mode_logits"] = mode_logits
            output["mode_probs"] = mode_probabilities
            output["reliability"] = reliability

        if shared_flat is not None:
            output["routed_output"] = routed_flat.reshape(batch_size, num_modes, self.hidden_dim)
            output["shared_output"] = shared_flat.reshape(batch_size, num_modes, self.hidden_dim)
        relation_embeddings = self.build_relation_embeddings(
            premise_embeddings,
            hypothesis_embeddings,
        )
        if relation_embeddings is not None:
            aligned_premise, aligned_hypothesis = relation_embeddings
            output["aligned_premise_embeddings"] = aligned_premise
            output["aligned_hypothesis_embeddings"] = aligned_hypothesis
        return output

    @torch.no_grad()
    def build_counterfactual_utility_targets(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Build detached per-mode, per-expert final-loss reduction targets."""

        if self.routing_strategy != "counterfactual_utility":
            raise RuntimeError(
                "Counterfactual utility targets require "
                "routing_strategy='counterfactual_utility'."
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"labels must have shape [{features.shape[0]}], got {tuple(labels.shape)}."
            )

        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            teacher_outputs = self(features)
            targets = self._build_counterfactual_targets_from_teacher(
                teacher_outputs,
                labels,
            )
        finally:
            for module, was_training in training_states:
                module.training = was_training

        if targets.requires_grad:
            raise RuntimeError("Counterfactual utility targets must be detached.")
        if not torch.isfinite(targets).all():
            raise FloatingPointError(
                "Counterfactual utility targets contain non-finite values."
            )
        return targets

    def _build_counterfactual_targets_from_teacher(
        self,
        teacher_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        projected = teacher_outputs["pre_moe_output"]
        teacher_modes = teacher_outputs["moe_output"]
        batch_size, num_modes, hidden_dim = projected.shape
        num_experts = len(self.routed_experts)

        baseline_loss = F.cross_entropy(
            teacher_outputs["logits"],
            labels,
            reduction="none",
        )
        flat = projected.reshape(batch_size * num_modes, hidden_dim)
        routed_candidates = torch.stack(
            [expert(flat) for expert in self.routed_experts],
            dim=1,
        ).reshape(batch_size, num_modes, num_experts, hidden_dim)
        shared_output = teacher_outputs.get("shared_output")
        reliability = teacher_outputs.get("reliability")

        counterfactual_losses: list[torch.Tensor] = []
        expanded_labels = labels.unsqueeze(1).expand(-1, num_experts).reshape(-1)
        for mode_index in range(num_modes):
            candidate = (
                projected[:, mode_index].unsqueeze(1)
                + routed_candidates[:, mode_index]
            )
            if shared_output is not None:
                candidate = candidate + shared_output[:, mode_index].unsqueeze(1)
            candidate = self.output_norm(candidate)

            counterfactual_modes = teacher_modes.unsqueeze(1).expand(
                -1,
                num_experts,
                -1,
                -1,
            ).clone()
            counterfactual_modes[:, :, mode_index] = candidate
            counterfactual_modes = counterfactual_modes.reshape(
                batch_size * num_experts,
                num_modes,
                hidden_dim,
            )

            fusion_reliability: torch.Tensor | None = None
            if self.use_reliability_fusion:
                if reliability is None:
                    raise RuntimeError(
                        "Counterfactual fusion requires teacher reliability."
                    )
                fusion_reliability = reliability.unsqueeze(1).expand(
                    -1,
                    num_experts,
                    -1,
                ).reshape(batch_size * num_experts, num_modes)
            fused, _ = self.mode_pooling(
                counterfactual_modes,
                fusion_reliability,
            )
            counterfactual_logits = self.classifier(fused)
            mode_losses = F.cross_entropy(
                counterfactual_logits,
                expanded_labels,
                reduction="none",
            ).reshape(batch_size, num_experts)
            counterfactual_losses.append(mode_losses)

        losses = torch.stack(counterfactual_losses, dim=1)
        return baseline_loss[:, None, None] - losses
