from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from KLTN.source.models.common.contrastive import NLIRelationContrastiveLoss
from KLTN.source.models.common.losses import compute_mode_auxiliary_loss
from KLTN.source.training.schedulers import build_lr_scheduler
from KLTN.source.training.trainer import compute_training_losses


class TestLabelSmoothing(unittest.TestCase):
    def test_final_and_mode_cross_entropy_use_the_same_smoothing(self) -> None:
        labels = torch.tensor([0, 2])
        logits = torch.tensor([[2.0, 0.0, -1.0], [-0.5, 0.5, 1.5]])
        mode_logits = torch.randn(2, 4, 3)
        outputs = {
            "logits": logits,
            "mode_logits": mode_logits,
        }

        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.0,
            relation_contrastive_loss_coef=0.0,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(),
            use_relation_contrastive_loss=False,
            mode_auxiliary_loss_coef=0.2,
            use_mode_auxiliary_loss=True,
            label_smoothing=0.05,
        )

        expected_classification = F.cross_entropy(
            logits,
            labels,
            label_smoothing=0.05,
        )
        expanded_labels = labels[:, None].expand(-1, 4).reshape(-1)
        expected_mode = F.cross_entropy(
            mode_logits.reshape(-1, 3),
            expanded_labels,
            label_smoothing=0.05,
        )
        torch.testing.assert_close(
            losses["classification_loss"],
            expected_classification,
        )
        torch.testing.assert_close(losses["mode_auxiliary_loss"], expected_mode)
        torch.testing.assert_close(
            losses["total_loss"],
            expected_classification + 0.2 * expected_mode,
        )

    def test_zero_smoothing_preserves_standard_cross_entropy(self) -> None:
        mode_logits = torch.randn(3, 4, 3)
        labels = torch.tensor([0, 1, 2])

        actual = compute_mode_auxiliary_loss(mode_logits, labels)
        expected = F.cross_entropy(
            mode_logits.reshape(-1, 3),
            labels[:, None].expand(-1, 4).reshape(-1),
        )

        torch.testing.assert_close(actual, expected)

    def test_invalid_smoothing_is_rejected(self) -> None:
        mode_logits = torch.randn(1, 4, 3)
        labels = torch.tensor([0])
        for smoothing in (-0.01, 1.0):
            with self.subTest(label_smoothing=smoothing):
                with self.assertRaisesRegex(ValueError, "label_smoothing"):
                    compute_mode_auxiliary_loss(
                        mode_logits,
                        labels,
                        label_smoothing=smoothing,
                    )


class TestCosineWithWarmupScheduler(unittest.TestCase):
    def test_warmup_reaches_peak_then_cosine_decays_to_minimum(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_lr_scheduler(
            optimizer,
            {
                "scheduler_name": "cosine_with_warmup",
                "scheduler_warmup_ratio": 0.2,
                "scheduler_min_learning_rate": 0.01,
                "num_epochs": 2,
            },
            steps_per_epoch=5,
        )
        self.assertIsNotNone(scheduler)

        learning_rates: list[float] = []
        for _ in range(10):
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(learning_rates[0], 0.05)
        self.assertAlmostEqual(max(learning_rates), 0.1)
        self.assertAlmostEqual(learning_rates[-1], 0.01)
        self.assertLess(learning_rates[0], learning_rates[1])
        self.assertTrue(
            all(
                current >= following
                for current, following in zip(
                    learning_rates[1:],
                    learning_rates[2:],
                )
            )
        )

    def test_missing_scheduler_config_keeps_fixed_learning_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)

        scheduler = build_lr_scheduler(optimizer, {}, steps_per_epoch=0)

        self.assertIsNone(scheduler)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_single_decay_step_reaches_minimum_learning_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = build_lr_scheduler(
            optimizer,
            {
                "scheduler_name": "cosine_with_warmup",
                "scheduler_warmup_ratio": 0.5,
                "scheduler_min_learning_rate": 0.01,
                "num_epochs": 2,
            },
            steps_per_epoch=1,
        )

        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_invalid_scheduler_configuration_is_rejected(self) -> None:
        invalid_configs = (
            {"scheduler_name": "unknown", "num_epochs": 2},
            {
                "scheduler_name": "cosine_with_warmup",
                "scheduler_warmup_ratio": 1.0,
                "num_epochs": 2,
            },
            {
                "scheduler_name": "cosine_with_warmup",
                "scheduler_min_learning_rate": 0.2,
                "num_epochs": 2,
            },
            {
                "scheduler_name": "cosine_with_warmup",
                "scheduler_min_learning_rate": float("nan"),
                "num_epochs": 2,
            },
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                parameter = torch.nn.Parameter(torch.tensor(1.0))
                optimizer = torch.optim.SGD([parameter], lr=0.1)
                with self.assertRaises(ValueError):
                    build_lr_scheduler(optimizer, config, steps_per_epoch=1)

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        with self.assertRaisesRegex(ValueError, "steps_per_epoch"):
            build_lr_scheduler(
                optimizer,
                {
                    "scheduler_name": "cosine_with_warmup",
                    "num_epochs": 2,
                },
                steps_per_epoch=0,
            )


if __name__ == "__main__":
    unittest.main()
