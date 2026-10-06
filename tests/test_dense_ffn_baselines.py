from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.data.schemas import INPUT_MODE_NAMES, InputMode
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.pair_feature_builder import (
    build_four_mode_features,
)
from KLTN.source.feature_pipelines.sonar_feature_pipeline.save_features import (
    extract_batch_features,
)
from KLTN.source.models.common.contrastive import NLIRelationContrastiveLoss
from KLTN.source.models.common.expert import FeedForwardExpert
from KLTN.source.models.common.invariance import CrossModalNLIInvarianceLoss
from KLTN.source.models.common.router import ReliabilityAwareRouter, TopKRouter
from KLTN.source.models.dense_ffn.model import (
    DenseFFNAttentionNLIModel,
    DenseFFNFullNLIModel,
    DenseFFNMeanNLIModel,
)
from KLTN.source.pair_features import build_pair_feature, split_pair_feature
from KLTN.source.training.metrics import EpochMetrics
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.trainer import (
    Trainer,
    build_feature_loader,
    compute_training_losses,
    resolve_feature_source_paths,
    resolve_training_output_dir,
)
from KLTN.source.training.utils import count_parameters, load_yaml_config


class TestDenseFFNArchitectures(unittest.TestCase):
    def test_forward_contract_and_no_moe_objects(self) -> None:
        for name, model in _small_models().items():
            with self.subTest(model=name):
                outputs = model(_dummy_features())

                self.assertEqual(outputs["logits"].shape, (3, 3))
                self.assertEqual(outputs["fused"].shape, (3, 8))
                self.assertEqual(outputs["dense_output"].shape, (3, 4, 8))
                self.assertFalse(hasattr(model, "router"))
                self.assertFalse(hasattr(model, "routed_experts"))
                self.assertFalse(hasattr(model, "shared_experts"))
                self.assertNotIn("load_balancing_loss", outputs)
                self.assertNotIn("router_probs", outputs)
                self.assertEqual(
                    sum(
                        isinstance(module, FeedForwardExpert)
                        for module in model.modules()
                    ),
                    1,
                )
                self.assertFalse(
                    any(
                        isinstance(module, (TopKRouter, ReliabilityAwareRouter))
                        for module in model.modules()
                    )
                )
                self.assertFalse(
                    any(isinstance(module, nn.ModuleList) for module in model.modules())
                )

    def test_residual_and_pooling_formulas(self) -> None:
        features = _dummy_features()
        for name, model in _small_models().items():
            with self.subTest(model=name):
                model.eval()
                premise, hypothesis = split_pair_feature(
                    features,
                    expected_embedding_dim=4,
                )
                canonical = build_pair_feature(
                    premise,
                    hypothesis,
                    expected_embedding_dim=4,
                )
                h_pre = model.input_projection(canonical)
                expected_dense = model.output_norm(h_pre + model.dense_ffn(h_pre))
                outputs = model(features)

                torch.testing.assert_close(outputs["dense_output"], expected_dense)
                if name == "mean":
                    torch.testing.assert_close(
                        outputs["fused"],
                        expected_dense.mean(dim=1),
                    )
                else:
                    attention = outputs["mode_attention_weights"]
                    self.assertEqual(attention.shape, (3, 4))
                    torch.testing.assert_close(
                        attention.sum(dim=1),
                        torch.ones(3),
                    )

    def test_mean_and_attention_use_only_classification_loss(self) -> None:
        labels = torch.tensor([0, 1, 2])
        for name in ("mean", "attention"):
            with self.subTest(model=name):
                model = _small_models()[name]
                outputs = model(_dummy_features())
                losses = _compute_losses(outputs, labels)
                losses["total_loss"].backward()

                torch.testing.assert_close(
                    losses["total_loss"],
                    F.cross_entropy(outputs["logits"], labels),
                )
                torch.testing.assert_close(
                    losses["weighted_load_balancing_loss"],
                    torch.tensor(0.0),
                )
                self.assertTrue(_has_gradient(model.input_projection))
                self.assertTrue(_has_gradient(model.dense_ffn))
                self.assertTrue(_has_gradient(model.output_norm))
                self.assertTrue(_has_gradient(model.classifier))
                if name == "attention":
                    self.assertTrue(_has_gradient(model.mode_pooling))

    def test_full_output_loss_and_backward(self) -> None:
        model = _small_models()["full"]
        labels = torch.tensor([0, 1, 2])
        outputs = model(_dummy_features())
        losses = _compute_losses(
            outputs,
            labels,
            use_relation=True,
            use_mode=True,
            use_invariance=True,
        )
        expected = (
            losses["classification_loss"]
            + losses["weighted_relation_contrastive_loss"]
            + losses["weighted_mode_auxiliary_loss"]
            + losses["weighted_invariance_loss"]
        )
        losses["total_loss"].backward()

        torch.testing.assert_close(losses["total_loss"], expected)
        torch.testing.assert_close(
            losses["weighted_load_balancing_loss"],
            torch.tensor(0.0),
        )
        self.assertEqual(outputs["mode_logits"].shape, (3, 4, 3))
        self.assertEqual(outputs["mode_probs"].shape, (3, 4, 3))
        self.assertEqual(outputs["reliability"].shape, (3, 4))
        self.assertTrue(((outputs["reliability"] >= 0) & (outputs["reliability"] <= 1)).all())
        self.assertEqual(outputs["mode_attention_weights"].shape, (3, 4))
        self.assertEqual(outputs["aligned_premise_embeddings"].shape, (3, 4, 4))
        self.assertEqual(outputs["aligned_hypothesis_embeddings"].shape, (3, 4, 4))
        for module in (
            model.input_projection,
            model.dense_ffn,
            model.output_norm,
            model.mode_nli_head,
            model.mode_pooling,
            model.classifier,
            model.alignment_projection,
        ):
            self.assertTrue(_has_gradient(module))

    def test_relation_projection_is_separate_from_classification(self) -> None:
        model = _small_models()["full"]
        model.eval()
        features = _dummy_features()
        logits_before = model(features)["logits"]
        with torch.no_grad():
            for parameter in model.alignment_projection.parameters():
                parameter.add_(torch.randn_like(parameter))
        logits_after = model(features)["logits"]

        torch.testing.assert_close(logits_before, logits_after)


class TestDenseFFNTrainingIntegration(unittest.TestCase):
    def test_factory_and_default_parameter_counts(self) -> None:
        expected = {
            "dense_ffn_mean.yaml": (DenseFFNMeanNLIModel, 8_408_067),
            "dense_ffn_attention.yaml": (DenseFFNAttentionNLIModel, 9_458_691),
            "dense_ffn_full.yaml": (DenseFFNFullNLIModel, 10_518_070),
        }
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for file_name, (model_type, parameter_count) in expected.items():
            with self.subTest(config=file_name):
                config = load_yaml_config(config_dir / file_name)
                model = build_model(config)
                self.assertIsInstance(model, model_type)
                self.assertEqual(
                    count_parameters(model)["total_parameters"],
                    parameter_count,
                )

    def test_positive_balancing_coefficient_requires_router_output(self) -> None:
        outputs = _small_models()["mean"](_dummy_features())
        with self.assertRaisesRegex(ValueError, "no load_balancing_loss"):
            _compute_losses(
                outputs,
                torch.tensor([0, 1, 2]),
                aux_loss_coef=0.01,
            )

    def test_configs_cover_six_isolated_feature_source_runs(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        output_directories: set[Path] = set()
        for file_name in (
            "dense_ffn_mean.yaml",
            "dense_ffn_attention.yaml",
            "dense_ffn_full.yaml",
        ):
            config = load_yaml_config(config_dir / file_name)
            for source in ("sonar", "phobert_xlsr"):
                selected, paths = resolve_feature_source_paths(
                    config,
                    feature_source=source,
                )
                output_directory = resolve_training_output_dir(
                    config["output_dir"],
                    selected,
                )

                self.assertEqual(selected, source)
                self.assertEqual(set(paths), {"train", "dev", "test"})
                self.assertIn(config["model_name"], output_directory.parts)
                self.assertEqual(output_directory.name, source)
                output_directories.add(output_directory)

        self.assertEqual(len(output_directories), 6)

    def test_dense_metrics_omit_routing_and_disabled_auxiliary_losses(self) -> None:
        labels = torch.tensor([0, 1, 2])
        model = _small_models()["attention"]
        outputs = model(_dummy_features())
        losses = _compute_losses(outputs, labels)
        metrics = EpochMetrics(
            NLIRelationContrastiveLoss(),
            CrossModalNLIInvarianceLoss(),
            aux_loss_coef=0.0,
            relation_loss_coef=0.0,
            use_relation_loss=False,
            report_disabled_losses=False,
        )
        metrics.update(outputs, labels, losses)
        report = metrics.compute()

        self.assertIn("classification_loss", report)
        self.assertIn("total_loss", report)
        self.assertIn("attention_TT_mean", report)
        for forbidden_prefix in (
            "load_balancing",
            "relation_contrastive",
            "mode_auxiliary",
            "invariance",
            "router_entropy",
            "expert_usage",
        ):
            self.assertFalse(any(key.startswith(forbidden_prefix) for key in report))

    def test_both_pipeline_outputs_use_one_saved_feature_loader(self) -> None:
        sonar_features = extract_batch_features(
            {"input_pairs": _build_input_pairs(batch_size=2)},
            _FakeSonarEncoder(),
        )
        phobert_xlsr_tensor = build_four_mode_features(
            torch.randn(2, 1024),
            torch.randn(2, 1024),
            torch.randn(2, 1024),
            torch.randn(2, 1024),
        )
        source_features = {
            "sonar": sonar_features,
            "phobert_xlsr": {
                mode: phobert_xlsr_tensor[:, mode_index]
                for mode_index, mode in enumerate(INPUT_MODE_NAMES)
            },
        }
        loader_config = {"input_dim": 4096, "batch_size": 2, "num_workers": 0}

        with tempfile.TemporaryDirectory() as directory:
            for source_name, features_by_mode in source_features.items():
                path = Path(directory) / f"{source_name}.pt"
                torch.save(_feature_payload(features_by_mode), path)
                batch = next(
                    iter(build_feature_loader(path, loader_config, shuffle=False))
                )

                self.assertEqual(batch["features"].shape, (2, 4, 4096))
                for model in _small_models(input_dim=4096).values():
                    self.assertEqual(model(batch["features"])["logits"].shape, (2, 3))

    def test_shared_trainer_runs_all_dense_baselines_without_routing(self) -> None:
        features = _dummy_features(batch_size=4)
        features_by_mode = {
            mode: features[:, mode_index]
            for mode_index, mode in enumerate(INPUT_MODE_NAMES)
        }
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.pt"
            torch.save(_feature_payload(features_by_mode), feature_path)
            for name, model in _small_models().items():
                with self.subTest(model=name):
                    use_full_losses = name == "full"
                    config = {
                        "model_name": f"dense_ffn_{name}",
                        "feature_source": "synthetic",
                        "input_dim": 16,
                        "num_labels": 3,
                        "batch_size": 2,
                        "num_workers": 0,
                        "learning_rate": 1e-4,
                        "weight_decay": 0.01,
                        "aux_loss_coef": 0.0,
                        "use_relation_contrastive_loss": use_full_losses,
                        "relation_contrastive_margin": 0.5,
                        "relation_contrastive_loss_coef": (
                            0.1 if use_full_losses else 0.0
                        ),
                        "use_mode_auxiliary_loss": use_full_losses,
                        "mode_auxiliary_loss_coef": 0.2 if use_full_losses else 0.0,
                        "use_invariance_loss": use_full_losses,
                        "invariance_loss_coef": 0.1 if use_full_losses else 0.0,
                        "reliability_weighted_invariance": False,
                    }
                    loader = build_feature_loader(feature_path, config, shuffle=False)
                    trainer = Trainer(
                        model=model,
                        train_loader=loader,
                        dev_loader=None,
                        config=config,
                        output_dir=Path(directory) / name,
                    )
                    try:
                        report = trainer._run_epoch(training=True)
                    finally:
                        trainer.close()

                    self.assertIn("classification_loss", report)
                    self.assertNotIn("load_balancing_loss", report)
                    self.assertFalse(
                        any(key.startswith("router_entropy") for key in report)
                    )
                    self.assertEqual(
                        "attention_TT_mean" in report,
                        name != "mean",
                    )
                    self.assertEqual(
                        "mode_auxiliary_loss" in report,
                        use_full_losses,
                    )


class _FakeSonarEncoder:
    def encode_text(self, values: list[str]) -> torch.Tensor:
        return torch.ones(len(values), 1024)

    def encode_speech(self, values: list[str]) -> torch.Tensor:
        return torch.full((len(values), 1024), 2.0)


def _small_models(input_dim: int = 16) -> dict[str, nn.Module]:
    common = {
        "input_dim": input_dim,
        "num_modes": 4,
        "hidden_dim": 8,
        "expert_ffn_dim": 16,
        "num_labels": 3,
        "dropout": 0.0,
    }
    return {
        "mean": DenseFFNMeanNLIModel(**common),
        "attention": DenseFFNAttentionNLIModel(**common),
        "full": DenseFFNFullNLIModel(
            **common,
            alignment_hidden_dim=4,
            reliability_embedding_dim=2,
        ),
    }


def _dummy_features(batch_size: int = 3) -> torch.Tensor:
    return build_pair_feature(
        torch.randn(batch_size, 4, 4),
        torch.randn(batch_size, 4, 4),
        expected_embedding_dim=4,
    )


def _compute_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    aux_loss_coef: float = 0.0,
    use_relation: bool = False,
    use_mode: bool = False,
    use_invariance: bool = False,
) -> dict[str, torch.Tensor]:
    return compute_training_losses(
        outputs,
        labels,
        aux_loss_coef=aux_loss_coef,
        relation_contrastive_loss_coef=0.1 if use_relation else 0.0,
        relation_contrastive_criterion=NLIRelationContrastiveLoss(),
        use_relation_contrastive_loss=use_relation,
        mode_auxiliary_loss_coef=0.2 if use_mode else 0.0,
        use_mode_auxiliary_loss=use_mode,
        invariance_loss_coef=0.1 if use_invariance else 0.0,
        use_invariance_loss=use_invariance,
        invariance_criterion=CrossModalNLIInvarianceLoss(),
    )


def _build_input_pairs(batch_size: int) -> dict[str, dict[str, list[str]]]:
    text_premise = [f"premise {index}" for index in range(batch_size)]
    text_hypothesis = [f"hypothesis {index}" for index in range(batch_size)]
    speech_premise = [f"premise_{index}.wav" for index in range(batch_size)]
    speech_hypothesis = [f"hypothesis_{index}.wav" for index in range(batch_size)]
    values = {
        InputMode.TEXT_TEXT: (text_premise, text_hypothesis),
        InputMode.TEXT_SPEECH: (text_premise, speech_hypothesis),
        InputMode.SPEECH_TEXT: (speech_premise, text_hypothesis),
        InputMode.SPEECH_SPEECH: (speech_premise, speech_hypothesis),
    }
    return {
        mode.value: {
            "premise": premise,
            "hypothesis": hypothesis,
            "premise_modality": [mode.premise_modality.value] * batch_size,
            "hypothesis_modality": [mode.hypothesis_modality.value] * batch_size,
            "input_mode": [mode.value] * batch_size,
        }
        for mode, (premise, hypothesis) in values.items()
    }


def _feature_payload(
    features_by_mode: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    batch_size = next(iter(features_by_mode.values())).shape[0]
    return [
        {
            "id": f"sample-{index}",
            "sample_index": index,
            "label": ("entailment", "contradiction")[index % 2],
            "features": {
                mode: features_by_mode[mode][index].detach().cpu()
                for mode in INPUT_MODE_NAMES
            },
        }
        for index in range(batch_size)
    ]


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
