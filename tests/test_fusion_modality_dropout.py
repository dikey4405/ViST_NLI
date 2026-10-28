from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from KLTN.source.models.common.fusion_modality_dropout import (
    FusionModalityDropout,
    register_fusion_modality_dropout,
)
from KLTN.source.models.common.pooling import ModeAttentionPooling
from KLTN.source.models.deepseek_moe.fusion_modality_dropout import (
    FusionModalityDropout as LegacyFusionModalityDropout,
)
from KLTN.source.models.deepseek_moe.fusion_modality_dropout import (
    register_fusion_modality_dropout as legacy_register_fusion_modality_dropout,
)
from KLTN.source.training.model_factory import build_model
from KLTN.source.training.utils import count_parameters, load_yaml_config


MOE_MODEL_NAMES = (
    "conventional_moe",
    "fine_grained_moe",
    "deepseek_moe",
)
RANDOM_PATCH_PATH = (
    "KLTN.source.models.common.fusion_modality_dropout.torch.rand"
)


class TestFusionModalityDropout(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_deepseek_import_path_reexports_shared_implementation(self) -> None:
        self.assertIs(LegacyFusionModalityDropout, FusionModalityDropout)
        self.assertIs(
            legacy_register_fusion_modality_dropout,
            register_fusion_modality_dropout,
        )

    def test_all_moe_architectures_register_exactly_one_hook(self) -> None:
        for model_name in MOE_MODEL_NAMES:
            with self.subTest(model=model_name):
                model = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=True,
                        fusion_drop_tt_prob=0.30,
                        fusion_drop_tt_st_prob=0.10,
                    )
                )
                hooks = tuple(model.mode_pooling._forward_hooks.values())
                self.assertEqual(len(hooks), 1)
                self.assertIsInstance(hooks[0], FusionModalityDropout)

    def test_disabled_and_zero_probability_are_exact_noops(self) -> None:
        for model_name in MOE_MODEL_NAMES:
            with self.subTest(model=model_name):
                baseline = build_model(_small_config(model_name))
                explicitly_disabled = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=False,
                        fusion_drop_tt_prob=1.0,
                    )
                )
                explicitly_disabled.load_state_dict(
                    baseline.state_dict(),
                    strict=True,
                )
                baseline.eval()
                explicitly_disabled.eval()

                features = _dummy_features(batch_size=2)
                baseline_outputs = baseline(features)
                disabled_outputs = explicitly_disabled(features)
                for key in ("logits", "fused", "mode_attention_weights"):
                    torch.testing.assert_close(
                        baseline_outputs[key],
                        disabled_outputs[key],
                    )
                self.assertEqual(
                    len(explicitly_disabled.mode_pooling._forward_hooks),
                    0,
                )

                enabled_without_probabilities = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=True,
                    )
                )
                self.assertEqual(
                    len(enabled_without_probabilities.mode_pooling._forward_hooks),
                    0,
                )

        pooling = ModeAttentionPooling(hidden_dim=5)
        with patch(RANDOM_PATCH_PATH) as random_mock:
            handle = register_fusion_modality_dropout(
                pooling,
                num_modes=4,
                drop_tt_prob=0.0,
                drop_tt_st_prob=0.0,
            )
            pooling(torch.randn(2, 4, 5))
        self.assertIsNone(handle)
        random_mock.assert_not_called()
        self.assertEqual(len(pooling._forward_hooks), 0)

    def test_forced_masks_apply_to_all_moe_architectures(self) -> None:
        cases = (
            (1.0, 0.0, (0,)),
            (0.0, 1.0, (0, 2)),
        )
        for model_name in MOE_MODEL_NAMES:
            for drop_tt_prob, drop_tt_st_prob, masked_indices in cases:
                with self.subTest(
                    model=model_name,
                    drop_tt_prob=drop_tt_prob,
                    drop_tt_st_prob=drop_tt_st_prob,
                ):
                    model = build_model(
                        _small_config(
                            model_name,
                            fusion_modality_dropout_enabled=True,
                            fusion_drop_tt_prob=drop_tt_prob,
                            fusion_drop_tt_st_prob=drop_tt_st_prob,
                        )
                    )
                    model.train()
                    outputs = model(_dummy_features(batch_size=3))
                    weights = outputs["mode_attention_weights"]

                    torch.testing.assert_close(
                        weights[:, list(masked_indices)],
                        torch.zeros(3, len(masked_indices)),
                    )
                    torch.testing.assert_close(
                        weights.sum(dim=1),
                        torch.ones(3),
                    )
                    expected_fused = torch.sum(
                        weights.unsqueeze(-1) * outputs["moe_output"],
                        dim=1,
                    )
                    torch.testing.assert_close(outputs["fused"], expected_fused)

    def test_sampling_is_independent_for_each_sample(self) -> None:
        pooling = ModeAttentionPooling(hidden_dim=5)
        handle = register_fusion_modality_dropout(
            pooling,
            num_modes=4,
            drop_tt_prob=0.30,
            drop_tt_st_prob=0.10,
        )
        self.addCleanup(handle.remove)
        mode_embeddings = torch.randn(3, 4, 5)

        def fixed_samples(batch_size: int, *, device: torch.device) -> torch.Tensor:
            self.assertEqual(batch_size, 3)
            return torch.tensor([0.10, 0.35, 0.80], device=device)

        with patch(RANDOM_PATCH_PATH, side_effect=fixed_samples):
            _, weights = pooling(mode_embeddings)

        self.assertEqual(weights[0, 0].item(), 0.0)
        self.assertGreater(weights[0, 2].item(), 0.0)
        self.assertEqual(weights[1, 0].item(), 0.0)
        self.assertEqual(weights[1, 2].item(), 0.0)
        self.assertTrue((weights[2] > 0).all())
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))

    def test_eval_mode_is_identical_to_baseline_for_all_moe_models(self) -> None:
        for model_name in MOE_MODEL_NAMES:
            with self.subTest(model=model_name):
                baseline = build_model(_small_config(model_name))
                dropout_model = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=True,
                        fusion_drop_tt_prob=1.0,
                        fusion_drop_tt_st_prob=0.0,
                    )
                )
                dropout_model.load_state_dict(baseline.state_dict(), strict=True)
                baseline.eval()
                dropout_model.eval()
                features = _dummy_features(batch_size=2)

                with patch(RANDOM_PATCH_PATH) as random_mock:
                    baseline_outputs = baseline(features)
                    dropout_outputs = dropout_model(features)

                random_mock.assert_not_called()
                for key in ("logits", "fused", "mode_attention_weights"):
                    torch.testing.assert_close(
                        baseline_outputs[key],
                        dropout_outputs[key],
                    )

    def test_classification_gradient_reaches_kept_modes_for_all_moe_models(
        self,
    ) -> None:
        for model_name in MOE_MODEL_NAMES:
            with self.subTest(model=model_name):
                model = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=True,
                        fusion_drop_tt_prob=1.0,
                        fusion_drop_tt_st_prob=0.0,
                    )
                )
                model.train()
                outputs = model(_dummy_features(batch_size=3))
                outputs["moe_output"].retain_grad()

                F.cross_entropy(
                    outputs["logits"],
                    torch.tensor([0, 1, 2]),
                ).backward()

                mode_gradients = outputs["moe_output"].grad
                self.assertIsNotNone(mode_gradients)
                torch.testing.assert_close(
                    mode_gradients[:, 0],
                    torch.zeros_like(mode_gradients[:, 0]),
                )
                for mode_index in (1, 2, 3):
                    self.assertGreater(
                        mode_gradients[:, mode_index].abs().sum().item(),
                        0.0,
                    )

    def test_numerical_stability_for_batch_one_and_common_dtypes(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(device=device.type, dtype=dtype):
                    hook = FusionModalityDropout(
                        drop_tt_prob=0.0,
                        drop_tt_st_prob=1.0,
                    )
                    module = nn.Identity().to(device)
                    module.train()
                    mode_embeddings = torch.randn(
                        1,
                        4,
                        6,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    attention = torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0]],
                        device=device,
                        dtype=dtype,
                    )
                    original_fused = torch.sum(
                        attention.unsqueeze(-1) * mode_embeddings,
                        dim=1,
                    )

                    result = hook(
                        module,
                        (mode_embeddings,),
                        (original_fused, attention),
                    )
                    self.assertIsNotNone(result)
                    fused, weights = result
                    self.assertTrue(torch.isfinite(fused).all())
                    self.assertTrue(torch.isfinite(weights).all())
                    torch.testing.assert_close(
                        weights.float().sum(dim=1),
                        torch.ones(1, device=device),
                        rtol=1e-3,
                        atol=1e-3,
                    )
                    self.assertEqual(weights[0, 0].item(), 0.0)
                    self.assertEqual(weights[0, 2].item(), 0.0)
                    fused.float().sum().backward()
                    self.assertGreater(
                        mode_embeddings.grad[:, [1, 3]].abs().sum().item(),
                        0.0,
                    )

    def test_validation_rejects_invalid_configuration_and_dense_models(self) -> None:
        invalid_probabilities = ((-0.1, 0.0), (1.1, 0.0), (0.6, 0.5))
        for drop_tt_prob, drop_tt_st_prob in invalid_probabilities:
            with self.subTest(
                drop_tt_prob=drop_tt_prob,
                drop_tt_st_prob=drop_tt_st_prob,
            ):
                with self.assertRaises(ValueError):
                    FusionModalityDropout(
                        drop_tt_prob=drop_tt_prob,
                        drop_tt_st_prob=drop_tt_st_prob,
                    )

        with self.assertRaisesRegex(ValueError, "num_modes=4"):
            register_fusion_modality_dropout(
                ModeAttentionPooling(hidden_dim=5),
                num_modes=3,
            )
        with self.assertRaisesRegex(ValueError, "only for MoE"):
            build_model(
                {
                    "model_name": "dense_ffn_mean",
                    "fusion_modality_dropout_enabled": True,
                }
            )

    def test_checkpoint_contract_and_experiment_configs(self) -> None:
        expected_contracts = {
            "conventional_moe": (23_114_554, 48),
            "fine_grained_moe": (23_122_878, 64),
            "deepseek_moe": (25_222_078, 68),
        }
        common_differences = {
            "fusion_modality_dropout_enabled",
            "fusion_drop_tt_prob",
            "fusion_drop_tt_st_prob",
            "output_dir",
        }

        for model_name, (parameter_count, state_key_count) in expected_contracts.items():
            with self.subTest(model=model_name):
                baseline_config = load_yaml_config(f"{model_name}.yaml")
                experiment_config = load_yaml_config(
                    f"{model_name}_fusion_dropout.yaml"
                )
                all_keys = set(baseline_config) | set(experiment_config)
                actual_differences = {
                    key
                    for key in all_keys
                    if baseline_config.get(key) != experiment_config.get(key)
                }
                expected_differences = (
                    {"output_dir"}
                    if model_name == "deepseek_moe"
                    else set(common_differences)
                )
                self.assertEqual(actual_differences, expected_differences)
                self.assertEqual(experiment_config["label_smoothing"], 0.0)
                self.assertEqual(
                    experiment_config["scheduler_name"],
                    "cosine_with_warmup",
                )
                self.assertEqual(
                    experiment_config["scheduler_warmup_ratio"],
                    0.03,
                )
                self.assertEqual(
                    experiment_config["scheduler_min_learning_rate"],
                    0.000001,
                )
                sonar_paths = experiment_config["feature_sources"]["sonar"]
                self.assertTrue(
                    all("Features/sonar_16k/" in path for path in sonar_paths.values())
                )

                baseline = build_model(baseline_config)
                experiment = build_model(experiment_config)
                self.assertEqual(
                    count_parameters(baseline)["total_parameters"],
                    parameter_count,
                )
                self.assertEqual(
                    count_parameters(experiment)["total_parameters"],
                    parameter_count,
                )
                self.assertEqual(
                    list(baseline.state_dict()),
                    list(experiment.state_dict()),
                )
                self.assertEqual(len(experiment.state_dict()), state_key_count)
                experiment.load_state_dict(baseline.state_dict(), strict=True)

    def test_counterfactual_teacher_disables_and_restores_dropout(self) -> None:
        for model_name in MOE_MODEL_NAMES:
            with self.subTest(model=model_name):
                model = build_model(
                    _small_config(
                        model_name,
                        fusion_modality_dropout_enabled=True,
                        fusion_drop_tt_prob=1.0,
                        fusion_drop_tt_st_prob=0.0,
                    )
                )
                model.train()
                features = _dummy_features(batch_size=2)
                labels = torch.tensor([0, 1])

                with patch(
                    RANDOM_PATCH_PATH,
                    side_effect=AssertionError(
                        "teacher must not sample modality dropout"
                    ),
                ):
                    first_targets = model.build_counterfactual_utility_targets(
                        features,
                        labels,
                    )
                    second_targets = model.build_counterfactual_utility_targets(
                        features,
                        labels,
                    )

                torch.testing.assert_close(first_targets, second_targets)
                self.assertTrue(model.training)
                self.assertTrue(model.mode_pooling.training)
                with patch(
                    RANDOM_PATCH_PATH,
                    return_value=torch.zeros(features.shape[0]),
                ):
                    training_outputs = model(features)
                torch.testing.assert_close(
                    training_outputs["mode_attention_weights"][:, 0],
                    torch.zeros(features.shape[0]),
                )


def _small_config(model_name: str, **overrides: object) -> dict[str, object]:
    architecture = {
        "conventional_moe": {
            "num_routed_experts": 2,
            "num_shared_experts": 0,
            "routed_top_k": 1,
        },
        "fine_grained_moe": {
            "num_routed_experts": 4,
            "num_shared_experts": 0,
            "routed_top_k": 2,
        },
        "deepseek_moe": {
            "num_routed_experts": 4,
            "num_shared_experts": 1,
            "routed_top_k": 2,
        },
    }[model_name]
    config: dict[str, object] = {
        "model_name": model_name,
        "input_dim": 16,
        "num_modes": 4,
        "hidden_dim": 8,
        "expert_ffn_dim": 16,
        "num_labels": 3,
        "dropout": 0.0,
        "use_relation_contrastive_loss": False,
        "alignment_hidden_dim": 4,
        "routing_strategy": "counterfactual_utility",
        "use_reliability_routing": False,
        "use_reliability_fusion": True,
        "detach_reliability_for_fusion": True,
        "mode_embedding_dim": 4,
        "reliability_embedding_dim": 4,
        **architecture,
    }
    config.update(overrides)
    return config


def _dummy_features(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 4, 16)


if __name__ == "__main__":
    unittest.main()
