from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.models.common.contrastive import NLIRelationContrastiveLoss
from KLTN.source.models.common.invariance import CrossModalNLIInvarianceLoss
from KLTN.source.models.common.losses import compute_mode_auxiliary_loss
from KLTN.source.models.common.pooling import ModeAttentionPooling
from KLTN.source.models.common.reliability import EntropyReliabilityEstimator
from KLTN.source.models.common.router import ReliabilityAwareRouter, TopKRouter
from KLTN.source.models.common.routing_utils import compute_routing_statistics
from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.models.deepseek_moe.model import DeepSeekMoENLIModel
from KLTN.source.models.fine_grained_moe.model import FineGrainedMoENLIModel
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.metrics import EpochMetrics, summarize_mode_values
from KLTN.source.training.trainer import compute_training_losses


class TestEntropyReliability(unittest.TestCase):
    def test_range_shape_and_numerical_stability(self) -> None:
        estimator = EntropyReliabilityEstimator(num_classes=3)
        logits = torch.tensor(
            [
                [[1000.0, -1000.0, -1000.0]] * 4,
                [[0.0, 0.0, 0.0]] * 4,
            ]
        )

        reliability = estimator(logits)

        self.assertEqual(reliability.shape, (2, 4))
        self.assertTrue(torch.isfinite(reliability).all())
        self.assertTrue((reliability >= 0).all())
        self.assertTrue((reliability <= 1).all())

    def test_confident_prediction_is_more_reliable_than_uniform(self) -> None:
        estimator = EntropyReliabilityEstimator(num_classes=3)
        confident = estimator(torch.tensor([[[10.0, -5.0, -5.0]]]))
        uniform = estimator(torch.zeros(1, 1, 3))
        self.assertGreater(confident.item(), uniform.item())


class TestInvarianceAndModeLoss(unittest.TestCase):
    def test_identical_predictions_have_zero_invariance(self) -> None:
        criterion = CrossModalNLIInvarianceLoss()
        probabilities = torch.tensor([[[0.7, 0.2, 0.1]] * 4])
        loss = criterion(probabilities)
        statistics = criterion.loss_statistics(probabilities)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)
        self.assertEqual(statistics["denominator"].item(), 6.0)

    def test_different_predictions_have_positive_differentiable_invariance(self) -> None:
        criterion = CrossModalNLIInvarianceLoss()
        logits = torch.tensor(
            [[[8.0, -4.0, -4.0], [-4.0, 8.0, -4.0], [-4.0, -4.0, 8.0], [8.0, -4.0, -4.0]]],
            requires_grad=True,
        )
        loss = criterion(torch.softmax(logits, dim=-1))

        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_weighted_invariance_detaches_reliability_weights(self) -> None:
        criterion = CrossModalNLIInvarianceLoss()
        logits = torch.randn(2, 4, 3, requires_grad=True)
        reliability = torch.rand(2, 4, requires_grad=True)

        loss = criterion(
            torch.softmax(logits, dim=-1),
            reliability,
            reliability_weighted=True,
        )
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertIsNone(reliability.grad)

    def test_mode_auxiliary_loss_is_average_cross_entropy(self) -> None:
        mode_logits = torch.randn(3, 4, 3)
        labels = torch.tensor([0, 1, 2])
        actual = compute_mode_auxiliary_loss(mode_logits, labels)
        expected = torch.stack(
            [F.cross_entropy(mode_logits[:, mode], labels) for mode in range(4)]
        ).mean()
        torch.testing.assert_close(actual, expected)

    def test_total_loss_contains_all_five_weighted_objectives(self) -> None:
        labels = torch.tensor([0, 2])
        mode_logits = torch.randn(2, 4, 3)
        outputs = {
            "logits": torch.randn(2, 3),
            "load_balancing_loss": torch.tensor(0.3),
            "aligned_premise_embeddings": torch.randn(2, 4, 4),
            "aligned_hypothesis_embeddings": torch.randn(2, 4, 4),
            "mode_logits": mode_logits,
            "mode_probs": torch.softmax(mode_logits, dim=-1),
            "reliability": torch.rand(2, 4),
        }

        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.01,
            relation_contrastive_loss_coef=0.1,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(),
            use_relation_contrastive_loss=True,
            mode_auxiliary_loss_coef=0.2,
            use_mode_auxiliary_loss=True,
            invariance_loss_coef=0.1,
            use_invariance_loss=True,
            invariance_criterion=CrossModalNLIInvarianceLoss(),
        )
        expected = sum(
            losses[key]
            for key in (
                "classification_loss",
                "weighted_load_balancing_loss",
                "weighted_relation_contrastive_loss",
                "weighted_mode_auxiliary_loss",
                "weighted_invariance_loss",
            )
        )

        torch.testing.assert_close(losses["total_loss"], expected)


class TestReliabilityAwareComponents(unittest.TestCase):
    def test_router_keeps_topk_contract(self) -> None:
        router = ReliabilityAwareRouter(
            hidden_dim=8,
            num_experts=4,
            top_k=2,
            mode_embedding_dim=3,
            reliability_embedding_dim=2,
        )
        output = router(
            torch.randn(8, 8),
            torch.randn(8, 3),
            torch.rand(8),
        )

        self.assertEqual(output["router_logits"].shape, (8, 4))
        self.assertEqual(output["topk_indices"].shape, (8, 2))
        self.assertEqual(output["topk_weights"].shape, (8, 2))
        torch.testing.assert_close(output["topk_weights"].sum(dim=-1), torch.ones(8))

    def test_mode_ids_are_distinct_and_broadcast_in_fixed_order(self) -> None:
        model = _full_models()["conventional"]
        mode_ids, mode_embeddings = model._build_mode_condition(3, torch.device("cpu"))

        self.assertEqual(mode_ids.shape, (3, 4))
        self.assertEqual(mode_embeddings.shape, (3, 4, 4))
        torch.testing.assert_close(mode_ids[0], torch.arange(4))
        torch.testing.assert_close(mode_ids[0], mode_ids[2])

    def test_detached_router_reliability_does_not_update_mode_head(self) -> None:
        model = _full_models()["conventional"]
        outputs = model(_dummy_features(batch_size=2))

        outputs["router_logits"].square().mean().backward()

        self.assertFalse(_has_gradient(model.mode_nli_head))
        self.assertTrue(_has_gradient(model.mode_embedding))
        self.assertTrue(_has_gradient(model.router.reliability_embedding))

        model.zero_grad(set_to_none=True)
        outputs = model(_dummy_features(batch_size=2))
        supervised = compute_mode_auxiliary_loss(
            outputs["mode_logits"],
            torch.tensor([0, 2]),
        )
        invariant = CrossModalNLIInvarianceLoss()(outputs["mode_probs"])
        (supervised + invariant).backward()
        self.assertTrue(_has_gradient(model.mode_nli_head))

    def test_detached_fusion_reliability_does_not_update_mode_head(self) -> None:
        model = _full_models()["conventional"]
        outputs = model(_dummy_features(batch_size=2))

        outputs["mode_attention_weights"].square().mean().backward()

        self.assertFalse(_has_gradient(model.mode_nli_head))
        self.assertTrue(_has_gradient(model.mode_pooling.reliability_score))

    def test_attention_uses_reliability_as_a_condition_not_a_weight(self) -> None:
        pooling = ModeAttentionPooling(
            hidden_dim=2,
            use_reliability=True,
            reliability_hidden_dim=2,
        )
        mode_embeddings = torch.randn(1, 4, 2)
        reliability = torch.tensor([[0.0, 0.2, 0.8, 1.0]])

        pooled, attention = pooling(mode_embeddings, reliability)

        self.assertEqual(pooled.shape, (1, 2))
        self.assertEqual(attention.shape, (1, 4))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(1))
        self.assertFalse(torch.allclose(attention, reliability / reliability.sum(dim=1)))

    def test_baseline_flags_preserve_unconditioned_path(self) -> None:
        model = ConventionalMoENLIModel(
            input_dim=16,
            hidden_dim=8,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=16,
            dropout=0.0,
            use_relation_contrastive_loss=False,
        )
        outputs = model(_dummy_features(batch_size=2))

        self.assertIsInstance(model.router, TopKRouter)
        self.assertNotIsInstance(model.router, ReliabilityAwareRouter)
        self.assertIsNone(model.mode_nli_head)
        self.assertNotIn("mode_logits", outputs)
        self.assertNotIn("reliability", outputs)

    def test_baseline_checkpoint_loads_into_full_model_non_strictly(self) -> None:
        baseline = ConventionalMoENLIModel(
            input_dim=16,
            hidden_dim=8,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=16,
            dropout=0.0,
            use_relation_contrastive_loss=False,
        )
        full = _full_models()["conventional"]

        incompatible = full.load_state_dict(baseline.state_dict(), strict=False)

        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(any(key.startswith("mode_nli_head.") for key in incompatible.missing_keys))
        self.assertTrue(any(key.startswith("mode_embedding.") for key in incompatible.missing_keys))
        with self.assertRaises(RuntimeError):
            full.load_state_dict(baseline.state_dict(), strict=True)


class TestFullReliabilityAwareModel(unittest.TestCase):
    def test_forward_backward_for_all_variants(self) -> None:
        for name, model in _full_models().items():
            with self.subTest(model=name):
                features = _dummy_features(batch_size=3).requires_grad_()
                labels = torch.tensor([0, 1, 2])
                outputs = model(features)
                losses = compute_training_losses(
                    outputs,
                    labels,
                    aux_loss_coef=0.01,
                    relation_contrastive_loss_coef=0.0,
                    relation_contrastive_criterion=NLIRelationContrastiveLoss(),
                    use_relation_contrastive_loss=False,
                    mode_auxiliary_loss_coef=0.2,
                    use_mode_auxiliary_loss=True,
                    invariance_loss_coef=0.1,
                    use_invariance_loss=True,
                    invariance_criterion=CrossModalNLIInvarianceLoss(),
                )
                losses["total_loss"].backward()

                self.assertEqual(outputs["logits"].shape, (3, 3))
                self.assertEqual(outputs["mode_logits"].shape, (3, 4, 3))
                self.assertEqual(outputs["mode_probs"].shape, (3, 4, 3))
                self.assertEqual(outputs["reliability"].shape, (3, 4))
                self.assertTrue(_has_gradient(model.input_projection))
                self.assertTrue(_has_gradient(model.mode_nli_head))
                self.assertTrue(_has_gradient(model.mode_embedding))
                self.assertTrue(_has_gradient(model.router))
                self.assertTrue(_has_gradient(model.router.reliability_embedding))
                self.assertTrue(_has_gradient(model.mode_pooling))
                self.assertTrue(_has_gradient(model.classifier))
                for expert_id in outputs["topk_indices"].unique().tolist():
                    self.assertTrue(_has_gradient(model.routed_experts[expert_id]))
                if isinstance(model, DeepSeekMoENLIModel):
                    self.assertTrue(all(_has_gradient(expert) for expert in model.shared_experts))

    def test_metrics_and_routing_statistics_are_normalized(self) -> None:
        model = _full_models()["conventional"]
        labels = torch.tensor([0, 1, 2])
        outputs = model(_dummy_features(batch_size=3))
        outputs["logits"] = torch.tensor(
            [[9.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 9.0]]
        )
        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.01,
            relation_contrastive_loss_coef=0.0,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(),
            use_relation_contrastive_loss=False,
            mode_auxiliary_loss_coef=0.2,
            use_mode_auxiliary_loss=True,
            invariance_loss_coef=0.1,
            use_invariance_loss=True,
            invariance_criterion=CrossModalNLIInvarianceLoss(),
        )
        metrics = EpochMetrics(
            NLIRelationContrastiveLoss(),
            CrossModalNLIInvarianceLoss(),
            aux_loss_coef=0.01,
            relation_loss_coef=0.0,
            use_relation_loss=False,
            mode_loss_coef=0.2,
            use_mode_loss=True,
            invariance_loss_coef=0.1,
            use_invariance_loss=True,
        )
        metrics.update(outputs, labels, losses)
        report = metrics.compute()
        routing = compute_routing_statistics(
            outputs["router_probs"],
            outputs["topk_indices"],
            outputs["topk_weights"],
        )

        for key in (
            "classification_loss",
            "mode_auxiliary_loss",
            "invariance_loss",
            "relation_contrastive_loss",
            "load_balancing_loss",
            "total_loss",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "reliability_TT_mean",
            "attention_SS_mean",
            "accuracy_TS",
            "router_entropy_ST",
        ):
            self.assertIn(key, report)
        torch.testing.assert_close(
            routing["expert_usage_by_mode"].sum(dim=1),
            torch.ones(4),
        )
        self.assertEqual(routing["router_entropy_by_mode"].shape, (4,))
        self.assertAlmostEqual(report["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(report["macro_f1"], 5.0 / 9.0)
        self.assertAlmostEqual(report["weighted_f1"], 5.0 / 9.0)

    def test_reliability_summary_reports_each_mode(self) -> None:
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        summary = summarize_mode_values(values, prefix="reliability")
        self.assertEqual(len(summary), 12)
        self.assertIn("reliability_TT_mean", summary)
        self.assertIn("reliability_SS_median", summary)


def _full_models() -> dict[str, nn.Module]:
    common = {
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "expert_ffn_dim": 16,
        "dropout": 0.0,
        "use_relation_contrastive_loss": False,
        "alignment_hidden_dim": 4,
        "use_mode_evidence": True,
        "use_reliability_routing": True,
        "use_reliability_fusion": True,
        "detach_reliability_for_routing": True,
        "detach_reliability_for_fusion": True,
        "mode_embedding_dim": 4,
        "reliability_embedding_dim": 2,
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


def _dummy_features(batch_size: int) -> torch.Tensor:
    premise = torch.randn(batch_size, 4, 4)
    hypothesis = torch.randn(batch_size, 4, 4)
    return build_pair_feature(premise, hypothesis, expected_embedding_dim=4)


def _has_gradient(module: nn.Module | None) -> bool:
    if module is None:
        return False
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


if __name__ == "__main__":
    unittest.main()
