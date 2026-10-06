from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ...pair_features import PAIR_FEATURE_COMPONENTS, build_pair_feature, split_pair_feature
from ..common.contrastive import PairAlignmentProjection
from ..common.expert import FeedForwardExpert, InputProjection, ModeNLIHead, NLIClassifier
from ..common.pooling import ModeAttentionPooling
from ..common.reliability import EntropyReliabilityEstimator
from ..common.routing_utils import validate_feature_tensor


PoolingType = Literal["mean", "attention"]


class _DenseFFNNLIModel(nn.Module):
    """Shared implementation for controlled non-MoE NLI baselines."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_modes: int,
        hidden_dim: int,
        expert_ffn_dim: int,
        num_labels: int,
        dropout: float,
        pooling: PoolingType,
        use_mode_evidence: bool = False,
        use_reliability_fusion: bool = False,
        reliability_estimator: str = "entropy",
        detach_reliability_for_fusion: bool = True,
        reliability_embedding_dim: int = 16,
        use_relation_contrastive_loss: bool = False,
        alignment_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if input_dim % PAIR_FEATURE_COMPONENTS != 0:
            raise ValueError(
                f"input_dim must be divisible by {PAIR_FEATURE_COMPONENTS}, got {input_dim}."
            )
        if pooling not in ("mean", "attention"):
            raise ValueError(f"Unsupported pooling type: {pooling!r}.")
        if use_reliability_fusion and not use_mode_evidence:
            raise ValueError("Reliability-aware fusion requires the shared mode evidence head.")
        if reliability_estimator != "entropy":
            raise ValueError(
                "Only entropy reliability is supported, "
                f"got '{reliability_estimator}'."
            )

        self.input_dim = input_dim
        self.embedding_dim = input_dim // PAIR_FEATURE_COMPONENTS
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_reliability_fusion = use_reliability_fusion
        self.detach_reliability_for_fusion = detach_reliability_for_fusion

        self.alignment_projection = (
            PairAlignmentProjection(self.embedding_dim, alignment_hidden_dim, dropout)
            if use_relation_contrastive_loss
            else None
        )
        self.input_projection = InputProjection(input_dim, hidden_dim, dropout)
        self.mode_nli_head = (
            ModeNLIHead(hidden_dim, num_labels, dropout)
            if use_mode_evidence
            else None
        )
        self.reliability_estimator = (
            EntropyReliabilityEstimator(num_labels)
            if use_mode_evidence
            else None
        )
        self.dense_ffn = FeedForwardExpert(hidden_dim, expert_ffn_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mode_pooling = (
            ModeAttentionPooling(
                hidden_dim,
                use_reliability=use_reliability_fusion,
                reliability_hidden_dim=reliability_embedding_dim,
            )
            if pooling == "attention"
            else None
        )
        self.classifier = NLIClassifier(hidden_dim, num_labels, dropout)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_feature_tensor(
            features,
            input_dim=self.input_dim,
            num_modes=self.num_modes,
        )
        premise_embeddings, hypothesis_embeddings = split_pair_feature(
            features,
            expected_embedding_dim=self.embedding_dim,
        )
        classification_features = build_pair_feature(
            premise_embeddings,
            hypothesis_embeddings,
            expected_embedding_dim=self.embedding_dim,
        )
        h_pre = self.input_projection(classification_features)

        mode_logits: torch.Tensor | None = None
        mode_probabilities: torch.Tensor | None = None
        reliability: torch.Tensor | None = None
        if self.mode_nli_head is not None and self.reliability_estimator is not None:
            mode_logits = self.mode_nli_head(h_pre)
            mode_probabilities = torch.softmax(mode_logits, dim=-1)
            reliability = self.reliability_estimator(mode_logits)

        dense_output = self.output_norm(h_pre + self.dense_ffn(h_pre))
        mode_attention_weights: torch.Tensor | None = None
        if self.mode_pooling is None:
            fused = dense_output.mean(dim=1)
        else:
            fusion_reliability = reliability
            if fusion_reliability is not None and self.detach_reliability_for_fusion:
                fusion_reliability = fusion_reliability.detach()
            fused, mode_attention_weights = self.mode_pooling(
                dense_output,
                fusion_reliability if self.use_reliability_fusion else None,
            )

        output = {
            "logits": self.classifier(fused),
            "fused": fused,
            "dense_output": dense_output,
        }
        if mode_attention_weights is not None:
            output["mode_attention_weights"] = mode_attention_weights
        if mode_logits is not None and mode_probabilities is not None and reliability is not None:
            output["mode_logits"] = mode_logits
            output["mode_probs"] = mode_probabilities
            output["reliability"] = reliability

        if self.alignment_projection is not None:
            output["aligned_premise_embeddings"] = self.alignment_projection(
                premise_embeddings
            )
            output["aligned_hypothesis_embeddings"] = self.alignment_projection(
                hypothesis_embeddings
            )
        return output


class DenseFFNMeanNLIModel(_DenseFFNNLIModel):
    """One shared dense FFN followed by arithmetic mean mode pooling."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        num_modes: int = 4,
        hidden_dim: int = 1024,
        expert_ffn_dim: int = 2048,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim,
            expert_ffn_dim=expert_ffn_dim,
            num_labels=num_labels,
            dropout=dropout,
            pooling="mean",
        )


class DenseFFNAttentionNLIModel(_DenseFFNNLIModel):
    """One shared dense FFN followed by semantic mode attention."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        num_modes: int = 4,
        hidden_dim: int = 1024,
        expert_ffn_dim: int = 2048,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim,
            expert_ffn_dim=expert_ffn_dim,
            num_labels=num_labels,
            dropout=dropout,
            pooling="attention",
        )


class DenseFFNFullNLIModel(_DenseFFNNLIModel):
    """Dense ablation retaining reliability, invariance, and relation branches."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        num_modes: int = 4,
        hidden_dim: int = 1024,
        expert_ffn_dim: int = 2048,
        num_labels: int = 3,
        dropout: float = 0.2,
        use_relation_contrastive_loss: bool = True,
        alignment_hidden_dim: int = 512,
        use_reliability_fusion: bool = True,
        reliability_estimator: str = "entropy",
        detach_reliability_for_fusion: bool = True,
        reliability_embedding_dim: int = 16,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim,
            expert_ffn_dim=expert_ffn_dim,
            num_labels=num_labels,
            dropout=dropout,
            pooling="attention",
            use_mode_evidence=True,
            use_reliability_fusion=use_reliability_fusion,
            reliability_estimator=reliability_estimator,
            detach_reliability_for_fusion=detach_reliability_for_fusion,
            reliability_embedding_dim=reliability_embedding_dim,
            use_relation_contrastive_loss=use_relation_contrastive_loss,
            alignment_hidden_dim=alignment_hidden_dim,
        )
