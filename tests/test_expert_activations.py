from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.models.common.expert import (
    FeedForwardExpert,
    SwiGLU,
    normalize_expert_activation,
    parameter_matched_swiglu_dim,
)
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.trainer import Trainer
from KLTN.source.training.utils import count_parameters, load_yaml_config


MOE_CONFIGS = (
    "conventional_moe.yaml",
    "conventional_moe_fusion_dropout.yaml",
    "fine_grained_moe.yaml",
    "fine_grained_moe_fusion_dropout.yaml",
    "deepseek_moe.yaml",
    "deepseek_moe_fusion_dropout.yaml",
    "deepseek_moe_post_moe_evidence.yaml",
    "deepseek_moe_post_moe_evidence_detached.yaml",
)

DENSE_CONFIGS = (
    "dense_ffn_mean.yaml",
    "dense_ffn_attention.yaml",
    "dense_ffn_full.yaml",
)


class TestFeedForwardExpertActivations(unittest.TestCase):
    def test_activation_names_are_case_insensitive_and_validated(self) -> None:
        expected = {
            " ReLU ": "relu",
            "gelu": "gelu",
            "SiLU": "silu",
            "SWIGLU": "swiglu",
        }
        for value, normalized in expected.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_expert_activation(value), normalized)

        with self.assertRaisesRegex(ValueError, "must be a string"):
            normalize_expert_activation(1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Expected one of"):
            normalize_expert_activation("tanh")

    def test_relu_gelu_and_silu_keep_the_standard_ffn_contract(self) -> None:
        activation_types = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        for name, activation_type in activation_types.items():
            with self.subTest(activation=name):
                expert = FeedForwardExpert(8, 12, 0.0, name)
                self.assertIsInstance(expert.net[1], activation_type)
                self.assertEqual(expert.effective_ffn_dim, 12)
                self.assertEqual(expert.net[0].out_features, 12)
                self.assertEqual(expert.net[3].in_features, 12)

                inputs = torch.randn(2, 3, 8, requires_grad=True)
                outputs = expert(inputs)
                self.assertEqual(outputs.shape, inputs.shape)
                self.assertTrue(torch.isfinite(outputs).all())
                outputs.square().mean().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(
                    all(parameter.grad is not None for parameter in expert.parameters())
                )

    def test_swiglu_formula_width_shape_and_gradient(self) -> None:
        projected = torch.randn(2, 10)
        value, gate = projected.chunk(2, dim=-1)
        self.assertTrue(torch.equal(SwiGLU()(projected), value * F.silu(gate)))
        with self.assertRaisesRegex(ValueError, "must be even"):
            SwiGLU()(torch.randn(2, 9))

        self.assertEqual(parameter_matched_swiglu_dim(1024), 683)
        self.assertEqual(parameter_matched_swiglu_dim(2048), 1365)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            parameter_matched_swiglu_dim(0)

        expert = FeedForwardExpert(8, 12, 0.0, "swiglu")
        self.assertIsInstance(expert.net[1], SwiGLU)
        self.assertEqual(expert.effective_ffn_dim, 8)
        self.assertEqual(expert.net[0].out_features, 16)
        self.assertEqual(expert.net[3].in_features, 8)

        inputs = torch.randn(2, 3, 8, requires_grad=True)
        outputs = expert(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(torch.isfinite(outputs).all())
        outputs.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in expert.parameters())
        )

    def test_default_gelu_preserves_state_and_output(self) -> None:
        torch.manual_seed(7)
        default = FeedForwardExpert(8, 12, 0.0)
        torch.manual_seed(7)
        explicit = FeedForwardExpert(8, 12, 0.0, "GeLU")

        self.assertEqual(list(default.state_dict()), list(explicit.state_dict()))
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))

        inputs = torch.randn(2, 4, 8)
        self.assertTrue(torch.equal(default(inputs), explicit(inputs)))


class TestMoEActivationIntegration(unittest.TestCase):
    def test_all_four_activations_forward_and_backward_for_each_moe(self) -> None:
        for model_name in (
            "conventional_moe",
            "fine_grained_moe",
            "deepseek_moe",
        ):
            for activation in ("relu", "gelu", "silu", "swiglu"):
                with self.subTest(model=model_name, activation=activation):
                    model = build_model(_small_moe_config(model_name, activation))
                    experts = list(model.routed_experts) + list(model.shared_experts)
                    self.assertTrue(experts)
                    self.assertTrue(
                        all(
                            expert.activation_function == activation
                            for expert in experts
                        )
                    )
                    self.assertIsInstance(model.input_projection.net[2], nn.GELU)

                    features = torch.randn(2, 4, 16)
                    outputs = model(features)
                    self.assertEqual(outputs["logits"].shape, (2, 3))
                    self.assertEqual(outputs["moe_output"].shape, (2, 4, 8))
                    outputs["logits"].square().mean().backward()
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            for expert in experts
                            for parameter in expert.parameters()
                        )
                    )

    def test_yaml_scope_and_dense_rejection(self) -> None:
        for file_name in MOE_CONFIGS:
            with self.subTest(config=file_name):
                self.assertEqual(
                    load_yaml_config(file_name)["activation_function"],
                    "gelu",
                )
        for file_name in DENSE_CONFIGS:
            with self.subTest(config=file_name):
                self.assertNotIn(
                    "activation_function",
                    load_yaml_config(file_name),
                )

        with self.assertRaisesRegex(ValueError, "supported only for MoE"):
            build_model(
                {
                    "model_name": "dense_ffn_mean",
                    "activation_function": "relu",
                }
            )

        legacy_config = _small_moe_config("conventional_moe", "gelu")
        del legacy_config["activation_function"]
        legacy_model = build_model(legacy_config)
        self.assertEqual(legacy_model.activation_function, "gelu")

    def test_parameter_matched_swiglu_counts_and_gelu_compatibility(self) -> None:
        expected = {
            "conventional_moe.yaml": (23_114_554, 23_113_186),
            "fine_grained_moe.yaml": (23_122_878, 23_133_806),
            "deepseek_moe.yaml": (51_469_246, 51_490_756),
        }
        for file_name, (gelu_count, swiglu_count) in expected.items():
            with self.subTest(config=file_name):
                config = load_yaml_config(file_name)
                gelu_model = build_model(config)
                gelu_keys = list(gelu_model.state_dict())
                self.assertEqual(
                    count_parameters(gelu_model)["total_parameters"],
                    gelu_count,
                )

                swiglu_config = dict(config)
                swiglu_config["activation_function"] = "swiglu"
                swiglu_model = build_model(swiglu_config)
                self.assertEqual(
                    count_parameters(swiglu_model)["total_parameters"],
                    swiglu_count,
                )
                self.assertEqual(gelu_keys, list(swiglu_model.state_dict()))

    def test_checkpoint_activation_validation_and_legacy_default(self) -> None:
        trainer = object.__new__(Trainer)
        trainer.activation_function = "gelu"
        trainer._validate_checkpoint_activation_function({"config": {}})
        trainer._validate_checkpoint_activation_function(
            {"config": {"activation_function": "GELU"}}
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            trainer._validate_checkpoint_activation_function(
                {"config": {"activation_function": "relu"}}
            )
        with self.assertRaisesRegex(ValueError, "missing config metadata"):
            trainer._validate_checkpoint_activation_function({})


def _small_moe_config(model_name: str, activation_function: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "num_routed_experts": 3,
        "num_shared_experts": 1 if model_name == "deepseek_moe" else 0,
        "routed_top_k": 2,
        "expert_ffn_dim": 12,
        "activation_function": activation_function,
        "num_labels": 3,
        "dropout": 0.0,
        "use_relation_contrastive_loss": False,
        "relation_contrastive_loss_coef": 0.0,
        "alignment_hidden_dim": 4,
        "routing_strategy": "topk",
        "use_reliability_routing": False,
        "use_reliability_fusion": False,
        "use_mode_auxiliary_loss": False,
        "use_invariance_loss": False,
    }


if __name__ == "__main__":
    unittest.main()
