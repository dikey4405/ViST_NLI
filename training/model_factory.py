from __future__ import annotations

from typing import Any

from torch import nn


COMMON_MODEL_KWARGS = (
    "input_dim",
    "num_modes",
    "hidden_dim",
    "expert_ffn_dim",
    "num_labels",
    "dropout",
)

MOE_MODEL_KWARGS = COMMON_MODEL_KWARGS + (
    "num_routed_experts",
    "num_shared_experts",
    "routed_top_k",
    "use_relation_contrastive_loss",
    "alignment_hidden_dim",
    "use_mode_evidence",
    "use_reliability_routing",
    "use_reliability_fusion",
    "reliability_estimator",
    "detach_reliability_for_routing",
    "detach_reliability_for_fusion",
    "mode_embedding_dim",
    "reliability_embedding_dim",
    "routing_strategy",
)

DENSE_FULL_MODEL_KWARGS = COMMON_MODEL_KWARGS + (
    "use_relation_contrastive_loss",
    "alignment_hidden_dim",
    "use_reliability_fusion",
    "reliability_estimator",
    "detach_reliability_for_fusion",
    "reliability_embedding_dim",
)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate exactly one model architecture from config."""

    model_name = config["model_name"]
    if model_name in {
        "dense_ffn_mean",
        "dense_ffn_attention",
        "dense_ffn_full",
    }:
        from ..models.dense_ffn.model import (
            DenseFFNAttentionNLIModel,
            DenseFFNFullNLIModel,
            DenseFFNMeanNLIModel,
        )

        dense_models = {
            "dense_ffn_mean": (DenseFFNMeanNLIModel, COMMON_MODEL_KWARGS),
            "dense_ffn_attention": (
                DenseFFNAttentionNLIModel,
                COMMON_MODEL_KWARGS,
            ),
            "dense_ffn_full": (DenseFFNFullNLIModel, DENSE_FULL_MODEL_KWARGS),
        }
        model_class, allowed_kwargs = dense_models[model_name]
        kwargs = {key: config[key] for key in allowed_kwargs if key in config}
        if (
            model_name == "dense_ffn_full"
            and float(config.get("relation_contrastive_loss_coef", 0.0)) == 0.0
        ):
            kwargs["use_relation_contrastive_loss"] = False
        return model_class(**kwargs)

    kwargs = {key: config[key] for key in MOE_MODEL_KWARGS if key in config}
    routing_strategy = config.get("routing_strategy")
    if routing_strategy is None:
        routing_strategy = (
            "reliability"
            if bool(config.get("use_reliability_routing", False))
            else "topk"
        )
    elif (
        bool(config.get("use_reliability_routing", False))
        and routing_strategy != "reliability"
    ):
        raise ValueError(
            "use_reliability_routing=true is only compatible with "
            "routing_strategy='reliability'."
        )
    reliability_routing = routing_strategy == "reliability"
    reliability_fusion = bool(config.get("use_reliability_fusion", False))
    mode_auxiliary = bool(config.get("use_mode_auxiliary_loss", False))
    invariance = bool(config.get("use_invariance_loss", False))
    kwargs["use_mode_evidence"] = any(
        (reliability_routing, reliability_fusion, mode_auxiliary, invariance)
    )
    if float(config.get("relation_contrastive_loss_coef", 0.0)) == 0.0:
        kwargs["use_relation_contrastive_loss"] = False

    if model_name == "conventional_moe":
        from ..models.conventional_moe.model import ConventionalMoENLIModel

        return ConventionalMoENLIModel(**kwargs)
    if model_name == "fine_grained_moe":
        from ..models.fine_grained_moe.model import FineGrainedMoENLIModel

        return FineGrainedMoENLIModel(**kwargs)
    if model_name == "deepseek_moe":
        from ..models.deepseek_moe.model import DeepSeekMoENLIModel

        return DeepSeekMoENLIModel(**kwargs)

    raise ValueError(f"Unsupported model_name: {model_name}")
