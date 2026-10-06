from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.models.common.pooling import ModeAttentionPooling
from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.models.deepseek_moe.model import DeepSeekMoENLIModel
from KLTN.source.models.fine_grained_moe.model import FineGrainedMoENLIModel
from KLTN.source.pair_features import build_pair_feature


class TestModeAttentionPooling(unittest.TestCase):
    def test_shapes_normalization_and_gradient(self) -> None:
        pooling = ModeAttentionPooling(hidden_dim=8)
        mode_embeddings = torch.randn(3, 4, 8, requires_grad=True)

        pooled, attention_weights = pooling(mode_embeddings)

        self.assertEqual(pooled.shape, (3, 8))
        self.assertEqual(attention_weights.shape, (3, 4))
        torch.testing.assert_close(
            attention_weights.sum(dim=1),
            torch.ones(3),
        )
        pooled.square().mean().backward()
        self.assertIsNotNone(mode_embeddings.grad)
        self.assertTrue(_has_gradient(pooling))

    def test_rejects_wrong_hidden_dimension(self) -> None:
        pooling = ModeAttentionPooling(hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "last dimension"):
            pooling(torch.randn(2, 4, 7))


class TestResidualMoEVariants(unittest.TestCase):
    def test_residual_formula_and_shape_for_all_variants(self) -> None:
        for name, model in _build_models().items():
            with self.subTest(model=name):
                flat = torch.randn(7, model.hidden_dim)
                routing = model.router(flat)

                moe_flat, routed_flat, shared_flat = model._apply_moe_block(
                    flat,
                    routing,
                )

                expected = flat + routed_flat
                if shared_flat is not None:
                    expected = expected + shared_flat
                expected = model.output_norm(expected)
                self.assertEqual(moe_flat.shape, flat.shape)
                torch.testing.assert_close(moe_flat, expected)

    def test_forward_backward_for_all_variants(self) -> None:
        for name, model in _build_models().items():
            with self.subTest(model=name):
                premise = torch.randn(3, 4, 4)
                hypothesis = torch.randn(3, 4, 4)
                features = build_pair_feature(
                    premise,
                    hypothesis,
                    expected_embedding_dim=4,
                ).requires_grad_()
                labels = torch.tensor([0, 1, 2])

                outputs = model(features)
                loss = (
                    F.cross_entropy(outputs["logits"], labels)
                    + 0.01 * outputs["load_balancing_loss"]
                )
                loss.backward()

                self.assertEqual(outputs["logits"].shape, (3, 3))
                self.assertEqual(outputs["moe_output"].shape, (3, 4, 8))
                self.assertEqual(outputs["mode_attention_weights"].shape, (3, 4))
                torch.testing.assert_close(
                    outputs["mode_attention_weights"].sum(dim=1),
                    torch.ones(3),
                )
                self.assertIsNotNone(features.grad)
                self.assertTrue(_has_gradient(model.input_projection))
                self.assertTrue(_has_gradient(model.router))
                self.assertTrue(_has_gradient(model.mode_pooling))
                self.assertTrue(_has_gradient(model.classifier))

                selected_experts = outputs["topk_indices"].unique().tolist()
                self.assertTrue(selected_experts)
                for expert_id in selected_experts:
                    self.assertTrue(_has_gradient(model.routed_experts[expert_id]))

                if isinstance(model, DeepSeekMoENLIModel):
                    self.assertTrue(all(_has_gradient(expert) for expert in model.shared_experts))

    def test_residual_branch_backpropagates_when_expert_output_is_zero(self) -> None:
        model = _build_models()["conventional"]
        with torch.no_grad():
            for expert in model.routed_experts:
                for parameter in expert.parameters():
                    parameter.zero_()
        flat = torch.randn(5, model.hidden_dim, requires_grad=True)
        routing = model.router(flat)

        moe_flat, routed_flat, _ = model._apply_moe_block(flat, routing)
        weighted_sum = (
            moe_flat * torch.arange(1, model.hidden_dim + 1, dtype=moe_flat.dtype)
        ).sum()
        weighted_sum.backward()

        torch.testing.assert_close(routed_flat, torch.zeros_like(routed_flat))
        self.assertIsNotNone(flat.grad)
        self.assertGreater(flat.grad.abs().sum().item(), 0.0)


def _build_models() -> dict[str, nn.Module]:
    common = {
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "expert_ffn_dim": 16,
        "dropout": 0.0,
        "use_relation_contrastive_loss": False,
        "alignment_hidden_dim": 4,
    }
    return {
        "conventional": ConventionalMoENLIModel(
            num_routed_experts=2,
            routed_top_k=1,
            **common,
        ),
        "fine_grained": FineGrainedMoENLIModel(
            num_routed_experts=4,
            routed_top_k=2,
            **common,
        ),
        "deepseek": DeepSeekMoENLIModel(
            num_routed_experts=4,
            num_shared_experts=1,
            routed_top_k=2,
            **common,
        ),
    }


def _has_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


if __name__ == "__main__":
    unittest.main()
