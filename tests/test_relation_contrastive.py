from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from KLTN.source.models.common.contrastive import NLIRelationContrastiveLoss
from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.trainer import compute_training_losses


class TestNLIRelationContrastiveLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = NLIRelationContrastiveLoss(margin=0.5)

    def test_entailment_is_pulled_together(self) -> None:
        premise = torch.tensor([[[1.0, 0.0]]])
        close_hypothesis = torch.tensor([[[1.0, 0.0]]])
        far_hypothesis = torch.tensor([[[-1.0, 0.0]]])
        labels = torch.tensor([0])

        close_loss = self.criterion(premise, close_hypothesis, labels)
        far_loss = self.criterion(premise, far_hypothesis, labels)

        self.assertLess(close_loss.item(), far_loss.item())

    def test_contradiction_is_pushed_beyond_margin(self) -> None:
        premise = torch.tensor([[[1.0, 0.0]]])
        close_hypothesis = torch.tensor([[[1.0, 0.0]]])
        far_hypothesis = torch.tensor([[[-1.0, 0.0]]])
        labels = torch.tensor([2])

        close_loss = self.criterion(premise, close_hypothesis, labels)
        far_loss = self.criterion(premise, far_hypothesis, labels)

        self.assertGreater(close_loss.item(), far_loss.item())
        self.assertEqual(far_loss.item(), 0.0)

    def test_neutral_samples_do_not_contribute(self) -> None:
        premise = torch.randn(2, 4, 8, requires_grad=True)
        hypothesis = torch.randn(2, 4, 8, requires_grad=True)

        loss = self.criterion(premise, hypothesis, torch.tensor([1, 1]))
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(premise.grad, torch.zeros_like(premise)))
        self.assertTrue(torch.equal(hypothesis.grad, torch.zeros_like(hypothesis)))


class TestRelationLossIntegration(unittest.TestCase):
    def test_total_loss_contains_all_three_terms(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0]], requires_grad=True)
        premise = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], requires_grad=True)
        hypothesis = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]], requires_grad=True)
        labels = torch.tensor([0, 2])
        load_balancing_loss = torch.tensor(1.5, requires_grad=True)
        criterion = NLIRelationContrastiveLoss(margin=0.5)
        outputs = {
            "logits": logits,
            "load_balancing_loss": load_balancing_loss,
            "aligned_premise_embeddings": premise,
            "aligned_hypothesis_embeddings": hypothesis,
        }

        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.01,
            relation_contrastive_loss_coef=0.1,
            relation_contrastive_criterion=criterion,
            use_relation_contrastive_loss=True,
        )
        expected = (
            F.cross_entropy(logits, labels)
            + 0.01 * load_balancing_loss
            + 0.1 * criterion(premise, hypothesis, labels)
        )

        self.assertTrue(torch.allclose(losses["total_loss"], expected))
        self.assertTrue(
            torch.allclose(losses["weighted_load_balancing_loss"], 0.01 * load_balancing_loss)
        )
        self.assertTrue(
            torch.allclose(
                losses["weighted_relation_contrastive_loss"],
                0.1 * losses["relation_contrastive_loss"],
            )
        )
        losses["total_loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(load_balancing_loss.grad)
        self.assertIsNotNone(premise.grad)
        self.assertIsNotNone(hypothesis.grad)

    def test_enabled_relation_loss_requires_alignment_outputs(self) -> None:
        outputs = {
            "logits": torch.randn(2, 3, requires_grad=True),
            "load_balancing_loss": torch.tensor(1.0, requires_grad=True),
        }

        with self.assertRaisesRegex(ValueError, "model output is missing"):
            compute_training_losses(
                outputs,
                torch.tensor([0, 2]),
                aux_loss_coef=0.01,
                relation_contrastive_loss_coef=0.1,
                relation_contrastive_criterion=NLIRelationContrastiveLoss(),
                use_relation_contrastive_loss=True,
            )

    def test_relation_projection_receives_auxiliary_gradient(self) -> None:
        torch.manual_seed(7)
        model = ConventionalMoENLIModel(
            input_dim=16,
            num_modes=4,
            hidden_dim=8,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=16,
            dropout=0.0,
            alignment_hidden_dim=4,
        )
        premise = torch.randn(2, 4, 4)
        hypothesis = torch.randn(2, 4, 4)
        features = build_pair_feature(premise, hypothesis, expected_embedding_dim=4)
        labels = torch.tensor([0, 2])

        outputs = model(features)
        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.01,
            relation_contrastive_loss_coef=0.1,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(margin=0.5),
            use_relation_contrastive_loss=True,
        )
        losses["total_loss"].backward()

        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["aligned_premise_embeddings"].shape, (2, 4, 4))
        self.assertEqual(outputs["aligned_hypothesis_embeddings"].shape, (2, 4, 4))
        self.assertTrue(_has_gradient(model.alignment_projection))

    def test_relation_projection_does_not_change_classification_branch(self) -> None:
        torch.manual_seed(11)
        model = ConventionalMoENLIModel(
            input_dim=16,
            num_modes=4,
            hidden_dim=8,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=16,
            dropout=0.0,
            alignment_hidden_dim=4,
        ).eval()
        premise = torch.randn(2, 4, 4)
        hypothesis = torch.randn(2, 4, 4)
        features = build_pair_feature(premise, hypothesis, expected_embedding_dim=4)

        expected_features = build_pair_feature(
            premise,
            hypothesis,
            expected_embedding_dim=4,
        )
        classification_features = model.build_classification_features(premise, hypothesis)
        outputs_before = model(features)

        with torch.no_grad():
            for parameter in model.alignment_projection.parameters():
                parameter.zero_()
        outputs_after = model(features)

        torch.testing.assert_close(classification_features, expected_features)
        torch.testing.assert_close(outputs_before["logits"], outputs_after["logits"])
        torch.testing.assert_close(outputs_before["moe_output"], outputs_after["moe_output"])
        self.assertFalse(
            torch.allclose(
                outputs_before["aligned_premise_embeddings"],
                outputs_after["aligned_premise_embeddings"],
            )
        )

    def test_disabled_relation_branch_is_not_required_by_training_loss(self) -> None:
        model = ConventionalMoENLIModel(
            input_dim=16,
            num_modes=4,
            hidden_dim=8,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=16,
            dropout=0.0,
            use_relation_contrastive_loss=False,
            alignment_hidden_dim=4,
        )
        premise = torch.randn(2, 4, 4)
        hypothesis = torch.randn(2, 4, 4)
        outputs = model(
            build_pair_feature(premise, hypothesis, expected_embedding_dim=4)
        )

        self.assertIsNone(model.alignment_projection)
        self.assertNotIn("aligned_premise_embeddings", outputs)
        self.assertNotIn("aligned_hypothesis_embeddings", outputs)
        losses = compute_training_losses(
            outputs,
            torch.tensor([0, 2]),
            aux_loss_coef=0.01,
            relation_contrastive_loss_coef=0.0,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(),
            use_relation_contrastive_loss=False,
        )
        losses["total_loss"].backward()
        self.assertTrue(_has_gradient(model.classifier))


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


if __name__ == "__main__":
    unittest.main()
