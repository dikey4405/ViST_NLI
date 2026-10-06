from __future__ import annotations

from ..common.base_model import RoutedMoENLIModel


class ConventionalMoENLIModel(RoutedMoENLIModel):
    """Conventional top-2 MoE baseline for multimodal NLI."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        num_modes: int = 4,
        hidden_dim: int = 1024,
        num_routed_experts: int = 4,
        num_shared_experts: int = 0,
        routed_top_k: int = 2,
        expert_ffn_dim: int = 2048,
        num_labels: int = 3,
        dropout: float = 0.2,
        use_relation_contrastive_loss: bool = True,
        alignment_hidden_dim: int = 512,
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
        if num_shared_experts != 0:
            raise ValueError("ConventionalMoENLIModel does not use shared experts.")

        super().__init__(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim,
            num_routed_experts=num_routed_experts,
            routed_top_k=routed_top_k,
            expert_ffn_dim=expert_ffn_dim,
            num_labels=num_labels,
            dropout=dropout,
            use_relation_contrastive_loss=use_relation_contrastive_loss,
            alignment_hidden_dim=alignment_hidden_dim,
            use_mode_evidence=use_mode_evidence,
            use_reliability_routing=use_reliability_routing,
            use_reliability_fusion=use_reliability_fusion,
            reliability_estimator=reliability_estimator,
            detach_reliability_for_routing=detach_reliability_for_routing,
            detach_reliability_for_fusion=detach_reliability_for_fusion,
            mode_embedding_dim=mode_embedding_dim,
            reliability_embedding_dim=reliability_embedding_dim,
            routing_strategy=routing_strategy,
        )
