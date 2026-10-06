from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.data.schemas import INPUT_MODE_NAMES
from KLTN.source.models.common.contrastive import NLIRelationContrastiveLoss
from KLTN.source.models.common.invariance import CrossModalNLIInvarianceLoss
from KLTN.source.models.common.router import (
    CounterfactualUtilityRouter,
    ReliabilityAwareRouter,
    TopKRouter,
)
from KLTN.source.models.conventional_moe.model import ConventionalMoENLIModel
from KLTN.source.models.deepseek_moe.model import DeepSeekMoENLIModel
from KLTN.source.models.fine_grained_moe.model import FineGrainedMoENLIModel
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.metrics import EpochMetrics
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.trainer import (
    Trainer,
    build_feature_loader,
    compute_training_losses,
)
from KLTN.source.training.utils import load_yaml_config


class TestCounterfactualUtilityRouter(unittest.TestCase):
    def test_contract_and_no_reliability_component(self) -> None:
        router = CounterfactualUtilityRouter(
            hidden_dim=8,
            num_experts=4,
            top_k=2,
            mode_embedding_dim=3,
        )

        outputs = router(torch.randn(8, 8), torch.randn(8, 3))

        self.assertEqual(outputs["router_logits"].shape, (8, 4))
        self.assertEqual(outputs["router_probs"].shape, (8, 4))
        self.assertEqual(outputs["topk_indices"].shape, (8, 2))
        self.assertEqual(outputs["topk_weights"].shape, (8, 2))
        torch.testing.assert_close(
            outputs["topk_weights"].sum(dim=-1),
            torch.ones(8),
        )
        self.assertFalse(hasattr(router, "reliability_embedding"))
        self.assertFalse(hasattr(router, "condition_router"))

    def test_strategy_resolution_and_legacy_support(self) -> None:
        common = _model_kwargs()
        topk = ConventionalMoENLIModel(**common)
        reliability = ConventionalMoENLIModel(
            **common,
            use_reliability_routing=True,
        )
        utility = ConventionalMoENLIModel(
            **common,
            routing_strategy="counterfactual_utility",
        )

        self.assertIsInstance(topk.router, TopKRouter)
        self.assertNotIsInstance(topk.router, ReliabilityAwareRouter)
        self.assertIsInstance(reliability.router, ReliabilityAwareRouter)
        self.assertIsInstance(utility.router, CounterfactualUtilityRouter)
        mode_ids, _ = utility._build_mode_condition(2, torch.device("cpu"))
        torch.testing.assert_close(mode_ids[0], torch.arange(4))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            ConventionalMoENLIModel(
                **common,
                routing_strategy="counterfactual_utility",
                use_reliability_routing=True,
            )

    def test_three_moe_configs_select_utility_routing_and_isolated_outputs(self) -> None:
        expected_outputs = {
            "conventional_moe": "outputs/conventional_moe_counterfactual_utility/",
            "fine_grained_moe": "outputs/fine_grained_moe_counterfactual_utility/",
            "deepseek_moe": "outputs/deepseek_moe_counterfactual_utility/",
        }
        for model_name, output_prefix in expected_outputs.items():
            with self.subTest(model=model_name):
                config = load_yaml_config(f"{model_name}.yaml")
                model = build_model(config)

                self.assertEqual(config["routing_strategy"], "counterfactual_utility")
                self.assertFalse(config["use_reliability_routing"])
                self.assertTrue(config["use_reliability_fusion"])
                self.assertEqual(config["counterfactual_routing_loss_coef"], 0.1)
                self.assertTrue(config["output_dir"].startswith(output_prefix))
                self.assertIsInstance(model.router, CounterfactualUtilityRouter)
                self.assertIsNotNone(model.mode_nli_head)
                self.assertIsNotNone(model.reliability_estimator)


class TestCounterfactualUtilityTargets(unittest.TestCase):
    def test_full_models_forward_target_and_backward(self) -> None:
        for name, model in _utility_models().items():
            with self.subTest(model=name):
                features = _dummy_features(batch_size=3).requires_grad_()
                labels = torch.tensor([0, 1, 2])

                outputs = model(features)
                self.assertNotIn("counterfactual_utility_targets", outputs)
                targets = model.build_counterfactual_utility_targets(features, labels)
                outputs["counterfactual_utility_targets"] = targets
                losses = _utility_losses(outputs, labels)
                losses["total_loss"].backward()

                num_experts = len(model.routed_experts)
                top_k = model.routed_top_k
                self.assertEqual(outputs["logits"].shape, (3, 3))
                self.assertEqual(outputs["mode_logits"].shape, (3, 4, 3))
                self.assertEqual(outputs["mode_probs"].shape, (3, 4, 3))
                self.assertEqual(outputs["reliability"].shape, (3, 4))
                self.assertEqual(outputs["router_logits"].shape, (3, 4, num_experts))
                self.assertEqual(outputs["topk_indices"].shape, (3, 4, top_k))
                self.assertEqual(outputs["topk_weights"].shape, (3, 4, top_k))
                self.assertEqual(outputs["pre_moe_output"].shape, (3, 4, 8))
                self.assertEqual(outputs["moe_output"].shape, (3, 4, 8))
                self.assertEqual(targets.shape, (3, 4, num_experts))
                self.assertFalse(targets.requires_grad)
                self.assertTrue(torch.isfinite(targets).all())
                self.assertTrue((outputs["reliability"] >= 0).all())
                self.assertTrue((outputs["reliability"] <= 1).all())
                self.assertTrue(_has_gradient(model.router))
                self.assertTrue(_has_gradient(model.mode_pooling))
                self.assertTrue(_has_gradient(model.classifier))
                self.assertTrue(_has_gradient(model.mode_nli_head))
                for expert_id in outputs["topk_indices"].unique().tolist():
                    self.assertTrue(_has_gradient(model.routed_experts[expert_id]))
                if isinstance(model, DeepSeekMoENLIModel):
                    self.assertTrue(
                        all(_has_gradient(expert) for expert in model.shared_experts)
                    )

    def test_target_generation_is_deterministic_and_restores_module_states(self) -> None:
        model = ConventionalMoENLIModel(
            **_model_kwargs(dropout=0.3),
            routing_strategy="counterfactual_utility",
            use_reliability_fusion=True,
        )
        model.train()
        model.mode_pooling.eval()
        states_before = [module.training for module in model.modules()]
        features = _dummy_features(batch_size=2)
        labels = torch.tensor([0, 2])

        first = model.build_counterfactual_utility_targets(features, labels)
        second = model.build_counterfactual_utility_targets(features, labels)

        torch.testing.assert_close(first, second)
        self.assertEqual(
            [module.training for module in model.modules()],
            states_before,
        )

    def test_better_expert_has_larger_signed_utility(self) -> None:
        model = ConventionalMoENLIModel(
            input_dim=16,
            num_modes=4,
            hidden_dim=2,
            num_routed_experts=2,
            routed_top_k=1,
            expert_ffn_dim=4,
            num_labels=3,
            dropout=0.0,
            use_relation_contrastive_loss=False,
            alignment_hidden_dim=2,
            routing_strategy="counterfactual_utility",
        )
        for parameter in model.input_projection.parameters():
            nn.init.zeros_(parameter)
        model.routed_experts = nn.ModuleList(
            [
                _ConstantExpert(torch.tensor([10.0, -10.0])),
                _ConstantExpert(torch.tensor([-10.0, 10.0])),
            ]
        )
        with torch.no_grad():
            model.router.router.weight.zero_()
            model.router.router.bias.copy_(torch.tensor([0.0, 10.0]))
            model.router.mode_utility.weight.zero_()
        model.mode_pooling = _UniformModePooling()
        classifier = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(
                torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
            )
        model.classifier = classifier

        targets = model.build_counterfactual_utility_targets(
            torch.zeros(1, 4, 16),
            torch.tensor([0]),
        )

        self.assertTrue(torch.all(targets[:, :, 0] > targets[:, :, 1]))
        self.assertTrue((targets[:, :, 0] > 0).all())

    def test_deepseek_counterfactual_candidates_keep_shared_output(self) -> None:
        model = DeepSeekMoENLIModel(
            **_model_kwargs(),
            num_shared_experts=1,
            routing_strategy="counterfactual_utility",
        )
        for parameter in model.shared_experts[0].parameters():
            nn.init.constant_(parameter, 0.1)
        recorder = _RecordingIdentity()
        model.output_norm = recorder
        features = _dummy_features(batch_size=1)
        labels = torch.tensor([0])
        model.eval()
        teacher = model(features)
        recorder.inputs.clear()

        model.build_counterfactual_utility_targets(features, labels)

        projected = teacher["pre_moe_output"]
        shared = teacher["shared_output"]
        flat = projected.reshape(4, model.hidden_dim)
        routed = torch.stack(
            [expert(flat) for expert in model.routed_experts],
            dim=1,
        ).reshape(1, 4, 2, model.hidden_dim)
        self.assertGreater(shared.abs().sum().item(), 0.0)
        self.assertEqual(len(recorder.inputs), 5)
        for mode_index in range(4):
            expected = (
                projected[:, mode_index].unsqueeze(1)
                + routed[:, mode_index]
                + shared[:, mode_index].unsqueeze(1)
            )
            torch.testing.assert_close(recorder.inputs[mode_index + 1], expected)

    def test_utility_loss_gradient_is_isolated_from_target_modules(self) -> None:
        model = _utility_models()["deepseek"]
        features = _dummy_features(batch_size=2)
        labels = torch.tensor([0, 2])
        outputs = model(features)
        outputs["counterfactual_utility_targets"] = (
            model.build_counterfactual_utility_targets(features, labels)
        )
        model.zero_grad(set_to_none=True)

        losses = compute_training_losses(
            outputs,
            labels,
            aux_loss_coef=0.0,
            relation_contrastive_loss_coef=0.0,
            relation_contrastive_criterion=NLIRelationContrastiveLoss(),
            use_relation_contrastive_loss=False,
            counterfactual_routing_loss_coef=1.0,
            use_counterfactual_routing_loss=True,
        )
        losses["counterfactual_routing_loss"].backward()

        self.assertTrue(_has_gradient(model.router))
        self.assertTrue(_has_gradient(model.mode_embedding))
        self.assertFalse(_has_gradient(model.routed_experts))
        self.assertFalse(_has_gradient(model.shared_experts))
        self.assertFalse(_has_gradient(model.mode_pooling))
        self.assertFalse(_has_gradient(model.classifier))
        self.assertFalse(_has_gradient(model.mode_nli_head))

    def test_utility_model_keeps_fusion_reliability_detached(self) -> None:
        model = _utility_models()["conventional"]
        outputs = model(_dummy_features(batch_size=2))

        outputs["mode_attention_weights"].square().mean().backward()

        self.assertFalse(_has_gradient(model.mode_nli_head))
        self.assertTrue(_has_gradient(model.mode_pooling.reliability_score))

    def test_total_loss_and_metrics_include_counterfactual_objective(self) -> None:
        model = _utility_models()["conventional"]
        labels = torch.tensor([0, 2])
        outputs = model(_dummy_features(batch_size=2))
        outputs["counterfactual_utility_targets"] = torch.randn_like(
            outputs["router_logits"]
        )
        losses = _utility_losses(outputs, labels)
        expected = sum(
            losses[key]
            for key in (
                "classification_loss",
                "weighted_load_balancing_loss",
                "weighted_mode_auxiliary_loss",
                "weighted_invariance_loss",
                "weighted_counterfactual_routing_loss",
            )
        )
        torch.testing.assert_close(losses["total_loss"], expected)
        torch.testing.assert_close(
            losses["counterfactual_routing_loss"],
            F.smooth_l1_loss(
                outputs["router_logits"],
                outputs["counterfactual_utility_targets"],
            ),
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
            counterfactual_routing_loss_coef=0.1,
            use_counterfactual_routing_loss=True,
        )
        metrics.update(outputs, labels, losses)
        report = metrics.compute()
        for key in (
            "counterfactual_routing_loss",
            "weighted_counterfactual_routing_loss",
            "counterfactual_utility_mean",
            "counterfactual_utility_std",
            "router_utility_top1_agreement",
        ):
            self.assertIn(key, report)
            self.assertTrue(torch.isfinite(torch.tensor(report[key])))


class TestCounterfactualTrainerIntegration(unittest.TestCase):
    def test_train_checkpoint_test_and_analysis_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.pt"
            output_dir = root / "output"
            torch.save(_feature_payload(), feature_path)
            config = _trainer_config()
            loader = build_feature_loader(feature_path, config, shuffle=False)
            model = _utility_models()["conventional"]
            trainer = Trainer(
                model=model,
                train_loader=loader,
                dev_loader=loader,
                config=config,
                output_dir=output_dir,
            )
            try:
                trainer.train()
                report = trainer.evaluate_test(loader)
                reliability = trainer.collect_reliability(loader)
                routing = trainer.collect_routing_statistics(loader)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    trainer._validate_checkpoint_routing_strategy(
                        {"config": {"routing_strategy": "reliability"}}
                    )
            finally:
                trainer.close()

            history = [
                json.loads(line)
                for line in (output_dir / "metrics_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            test_payload = json.loads(
                (output_dir / "test_metrics.json").read_text(encoding="utf-8")
            )
            log_text = (output_dir / "train.log").read_text(encoding="utf-8")

        self.assertEqual(len(history), 1)
        self.assertIn("counterfactual_routing_loss", history[0]["train"])
        self.assertIn("router_utility_top1_agreement", history[0]["dev"])
        self.assertIn("counterfactual_routing_loss", report["metrics"])
        self.assertEqual(test_payload["metrics"], report["metrics"])
        self.assertEqual(reliability.shape, (3, 4))
        self.assertEqual(len(routing), 1)
        self.assertIn("counterfactual_routing", log_text)
        self.assertNotIn("NaN", log_text)


class _ConstantExpert(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(inputs)


class _UniformModePooling(nn.Module):
    def forward(
        self,
        mode_embeddings: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del reliability
        weights = mode_embeddings.new_full(
            mode_embeddings.shape[:2],
            1.0 / mode_embeddings.shape[1],
        )
        return mode_embeddings.mean(dim=1), weights


class _RecordingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs.append(inputs.detach().clone())
        return inputs


def _model_kwargs(dropout: float = 0.0) -> dict[str, object]:
    return {
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "num_routed_experts": 2,
        "routed_top_k": 1,
        "expert_ffn_dim": 16,
        "num_labels": 3,
        "dropout": dropout,
        "use_relation_contrastive_loss": False,
        "alignment_hidden_dim": 4,
    }


def _utility_models() -> dict[str, nn.Module]:
    common = {
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
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
    return {
        "conventional": ConventionalMoENLIModel(
            **common,
            num_routed_experts=2,
            routed_top_k=1,
        ),
        "fine_grained": FineGrainedMoENLIModel(
            **common,
            num_routed_experts=3,
            routed_top_k=2,
        ),
        "deepseek": DeepSeekMoENLIModel(
            **common,
            num_routed_experts=3,
            num_shared_experts=1,
            routed_top_k=2,
        ),
    }


def _dummy_features(batch_size: int) -> torch.Tensor:
    return build_pair_feature(
        torch.randn(batch_size, 4, 4),
        torch.randn(batch_size, 4, 4),
        expected_embedding_dim=4,
    )


def _utility_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return compute_training_losses(
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
        counterfactual_routing_loss_coef=0.1,
        use_counterfactual_routing_loss=True,
    )


def _feature_payload() -> list[dict[str, object]]:
    features = _dummy_features(batch_size=3)
    labels = ("entailment", "neutral", "contradiction")
    return [
        {
            "id": f"sample-{index}",
            "sample_index": index,
            "label": labels[index],
            "features": {
                mode: features[index, mode_index]
                for mode_index, mode in enumerate(INPUT_MODE_NAMES)
            },
        }
        for index in range(3)
    ]


def _trainer_config() -> dict[str, object]:
    return {
        "model_name": "conventional_moe",
        "feature_source": "synthetic",
        "routing_strategy": "counterfactual_utility",
        "use_reliability_routing": False,
        "input_dim": 16,
        "num_labels": 3,
        "batch_size": 3,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "aux_loss_coef": 0.01,
        "use_relation_contrastive_loss": False,
        "relation_contrastive_margin": 0.5,
        "relation_contrastive_loss_coef": 0.0,
        "use_mode_auxiliary_loss": True,
        "mode_auxiliary_loss_coef": 0.2,
        "use_invariance_loss": True,
        "invariance_loss_coef": 0.1,
        "reliability_weighted_invariance": False,
        "counterfactual_routing_loss_coef": 0.1,
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
