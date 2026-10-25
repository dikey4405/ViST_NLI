from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from KLTN.source.models.common.invariance import CrossModalNLIInvarianceLoss
from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.models.deepseek_moe.model import DeepSeekMoENLIModel
from KLTN.source.models.fine_grained_moe.model import FineGrainedMoENLIModel
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.trainer import Trainer
from KLTN.source.training.utils import count_parameters, load_yaml_config


class TestPostMoEEvidenceArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_stage_validation_and_reliability_routing_conflict(self) -> None:
        for invalid_stage in ("before_projection", "after_fusion", "", None, 1):
            with self.subTest(stage=invalid_stage):
                with self.assertRaisesRegex(ValueError, "mode_evidence_stage"):
                    ConventionalMoENLIModel(
                        **_common_kwargs(),
                        mode_evidence_stage=invalid_stage,
                    )

        reliability_kwargs = _common_kwargs()
        reliability_kwargs["routing_strategy"] = "reliability"
        with self.assertRaisesRegex(ValueError, "requires reliability before"):
            ConventionalMoENLIModel(
                **reliability_kwargs,
                mode_evidence_stage="post_moe",
            )

    def test_evidence_uses_the_configured_representation_for_all_moe_models(self) -> None:
        features = _dummy_features(batch_size=2)
        for stage in ("pre_moe", "post_moe"):
            for name, model in _models(stage).items():
                with self.subTest(stage=stage, model=name):
                    model.eval()
                    with torch.no_grad():
                        outputs = model(features)
                        evidence_input = (
                            outputs["pre_moe_output"]
                            if stage == "pre_moe"
                            else outputs["moe_output"]
                        )
                        expected_logits = model.mode_nli_head(evidence_input)
                        expected_probabilities = torch.softmax(
                            expected_logits,
                            dim=-1,
                        )
                        expected_reliability = model.reliability_estimator(
                            expected_logits
                        )

                    torch.testing.assert_close(
                        outputs["mode_logits"], expected_logits, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        outputs["mode_probs"],
                        expected_probabilities,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        outputs["reliability"],
                        expected_reliability,
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(outputs["logits"].shape, (2, 3))
                    self.assertEqual(outputs["mode_logits"].shape, (2, 4, 3))
                    self.assertEqual(outputs["mode_probs"].shape, (2, 4, 3))
                    self.assertEqual(outputs["reliability"].shape, (2, 4))
                    self.assertEqual(
                        outputs["mode_attention_weights"].shape,
                        (2, 4),
                    )
                    self.assertTrue((outputs["reliability"] >= 0).all())
                    self.assertTrue((outputs["reliability"] <= 1).all())

    def test_mode_and_invariance_gradients_enter_moe_only_post_moe(self) -> None:
        labels = torch.tensor([0, 1, 2])
        criterion = CrossModalNLIInvarianceLoss()
        for objective_name in ("mode", "invariance"):
            for stage in ("pre_moe", "post_moe"):
                with self.subTest(objective=objective_name, stage=stage):
                    model = _models(stage)["deepseek"]
                    outputs = model(_dummy_features(batch_size=3))
                    if objective_name == "mode":
                        loss = F.cross_entropy(
                            outputs["mode_logits"].reshape(-1, 3),
                            labels.unsqueeze(1).expand(-1, 4).reshape(-1),
                        )
                    else:
                        loss = criterion(outputs["mode_probs"])
                    loss.backward()

                    self.assertTrue(_has_gradient(model.input_projection))
                    self.assertTrue(_has_gradient(model.mode_nli_head))
                    self.assertEqual(_has_gradient(model.router), stage == "post_moe")
                    self.assertEqual(
                        _has_gradient(model.mode_embedding),
                        stage == "post_moe",
                    )
                    self.assertEqual(
                        _has_gradient(model.output_norm),
                        stage == "post_moe",
                    )
                    for expert_id in outputs["topk_indices"].unique().tolist():
                        self.assertEqual(
                            _has_gradient(model.routed_experts[expert_id]),
                            stage == "post_moe",
                        )
                    for expert in model.shared_experts:
                        self.assertEqual(
                            _has_gradient(expert),
                            stage == "post_moe",
                        )
                    self.assertFalse(_has_gradient(model.mode_pooling))
                    self.assertFalse(_has_gradient(model.classifier))

    def test_post_moe_counterfactual_candidates_recompute_mode_evidence(self) -> None:
        features = _dummy_features(batch_size=2)
        labels = torch.tensor([0, 2])
        for stage, expected_calls in (("pre_moe", 1), ("post_moe", 5)):
            with self.subTest(stage=stage):
                model = _models(stage)["deepseek"]
                model.eval()
                with patch.object(
                    model.mode_nli_head,
                    "forward",
                    wraps=model.mode_nli_head.forward,
                ) as evidence_forward:
                    targets = model.build_counterfactual_utility_targets(
                        features,
                        labels,
                    )

                self.assertEqual(evidence_forward.call_count, expected_calls)
                self.assertEqual(targets.shape, (2, 4, 3))
                self.assertFalse(targets.requires_grad)
                self.assertTrue(torch.isfinite(targets).all())
                if stage == "post_moe":
                    for call in evidence_forward.call_args_list[1:]:
                        self.assertEqual(call.args[0].shape, (6, 4, 8))


class TestPostMoEEvidenceIntegration(unittest.TestCase):
    def test_yaml_is_controlled_and_keeps_the_checkpoint_contract(self) -> None:
        baseline_config = load_yaml_config("deepseek_moe_fusion_dropout.yaml")
        experiment_config = load_yaml_config("deepseek_moe_post_moe_evidence.yaml")
        all_keys = set(baseline_config) | set(experiment_config)
        differences = {
            key
            for key in all_keys
            if baseline_config.get(key) != experiment_config.get(key)
        }
        self.assertEqual(
            differences,
            {"mode_evidence_stage", "output_dir"},
        )
        self.assertEqual(baseline_config["mode_evidence_stage"], "pre_moe")
        self.assertEqual(experiment_config["mode_evidence_stage"], "post_moe")
        self.assertEqual(
            experiment_config["output_dir"],
            "outputs/deepseek_moe_post_moe_evidence/{feature_source}",
        )

        for model_name in (
            "conventional_moe",
            "fine_grained_moe",
            "deepseek_moe",
        ):
            for suffix in ("", "_fusion_dropout"):
                with self.subTest(model=model_name, suffix=suffix):
                    config = load_yaml_config(f"{model_name}{suffix}.yaml")
                    self.assertEqual(config["mode_evidence_stage"], "pre_moe")
        for dense_name in (
            "dense_ffn_mean",
            "dense_ffn_attention",
            "dense_ffn_full",
        ):
            self.assertNotIn(
                "mode_evidence_stage",
                load_yaml_config(f"{dense_name}.yaml"),
            )

        baseline = build_model(baseline_config)
        experiment = build_model(experiment_config)
        self.assertEqual(
            count_parameters(baseline)["total_parameters"],
            25_222_078,
        )
        self.assertEqual(
            count_parameters(experiment)["total_parameters"],
            25_222_078,
        )
        self.assertEqual(list(baseline.state_dict()), list(experiment.state_dict()))
        self.assertEqual(len(experiment.state_dict()), 68)
        experiment.load_state_dict(baseline.state_dict(), strict=True)

    def test_trainer_logs_stage_and_rejects_mismatched_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _models("post_moe")["deepseek"]
            features = _dummy_features(batch_size=2)
            loader = DataLoader(
                [
                    {"features": features[index], "labels": index}
                    for index in range(2)
                ],
                batch_size=2,
            )
            config = _trainer_config()
            trainer = Trainer(
                model=model,
                train_loader=loader,
                dev_loader=loader,
                config=config,
                output_dir=Path(directory) / "post_moe",
            )
            try:
                trainer.train()
                checkpoint = torch.load(
                    trainer.output_dir / "best_model.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    checkpoint["config"]["mode_evidence_stage"],
                    "post_moe",
                )
                with self.assertRaisesRegex(ValueError, "does not match"):
                    trainer._validate_checkpoint_mode_evidence_stage(
                        {
                            "config": {
                                "routing_strategy": "counterfactual_utility",
                                "mode_evidence_stage": "pre_moe",
                            }
                        }
                    )
                with self.assertRaisesRegex(ValueError, "does not match"):
                    trainer._validate_checkpoint_mode_evidence_stage(
                        {
                            "config": {
                                "routing_strategy": "counterfactual_utility",
                            }
                        }
                    )
            finally:
                trainer.close()

            log_text = (trainer.output_dir / "train.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("mode_evidence_stage=post_moe", log_text)
            self.assertEqual(
                Trainer._mode_evidence_stage_from_config({}),
                "pre_moe",
            )


def _common_kwargs() -> dict[str, object]:
    return {
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "num_routed_experts": 3,
        "routed_top_k": 2,
        "expert_ffn_dim": 16,
        "num_labels": 3,
        "dropout": 0.0,
        "use_relation_contrastive_loss": False,
        "alignment_hidden_dim": 4,
        "use_reliability_fusion": True,
        "detach_reliability_for_fusion": True,
        "routing_strategy": "counterfactual_utility",
        "mode_embedding_dim": 4,
        "reliability_embedding_dim": 2,
    }


def _models(stage: str) -> dict[str, nn.Module]:
    common = {**_common_kwargs(), "mode_evidence_stage": stage}
    return {
        "conventional": ConventionalMoENLIModel(**common),
        "fine_grained": FineGrainedMoENLIModel(**common),
        "deepseek": DeepSeekMoENLIModel(
            **common,
            num_shared_experts=1,
        ),
    }


def _dummy_features(batch_size: int) -> torch.Tensor:
    return build_pair_feature(
        torch.randn(batch_size, 4, 4),
        torch.randn(batch_size, 4, 4),
        expected_embedding_dim=4,
    )


def _trainer_config() -> dict[str, object]:
    return {
        "model_name": "deepseek_moe",
        "feature_source": "synthetic",
        "routing_strategy": "counterfactual_utility",
        "mode_evidence_stage": "post_moe",
        "input_dim": 16,
        "num_labels": 3,
        "batch_size": 2,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "aux_loss_coef": 0.0,
        "use_relation_contrastive_loss": False,
        "relation_contrastive_loss_coef": 0.0,
        "use_mode_auxiliary_loss": False,
        "mode_auxiliary_loss_coef": 0.0,
        "use_invariance_loss": False,
        "invariance_loss_coef": 0.0,
        "counterfactual_routing_loss_coef": 0.0,
        "num_epochs": 1,
        "patience": 1,
    }


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
