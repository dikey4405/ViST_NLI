from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from KLTN.source.data.schemas import INPUT_MODE_NAMES
from KLTN.source.models.dense_ffn.model import DenseFFNMeanNLIModel
from KLTN.source.training.logging_utils import (
    append_jsonl,
    format_epoch_summary,
    format_split_metrics,
)
from KLTN.source.training.trainer import Trainer, build_feature_loader


class TestTrainingLogFormatting(unittest.TestCase):
    def test_epoch_summary_has_stable_groups_and_precision(self) -> None:
        train_metrics = {
            "weighted_f1": 0.72,
            "classification_loss": 1.1,
            "macro_f1": 0.7,
            "total_loss": 1.23456789,
            "accuracy": 0.75,
            "relation_contrastive_loss": 0.2,
            "mode_auxiliary_loss": 0.3,
            "invariance_loss": 0.04,
            "load_balancing_loss": 0.01,
            "accuracy_TT": 0.74,
            "accuracy_TS": 0.73,
            "accuracy_ST": 0.72,
            "accuracy_SS": 0.71,
            "attention_TT_mean": 0.2,
            "attention_TS_mean": 0.3,
            "attention_ST_mean": 0.1,
            "attention_SS_mean": 0.4,
            "reliability_TT_mean": 0.9,
            "reliability_TS_mean": 0.8,
            "reliability_ST_mean": 0.7,
            "reliability_SS_mean": 0.6,
            "router_entropy_TT": 1.0,
            "expert_usage_TT_expert_1": 0.4,
            "expert_usage_TT_expert_0": 0.6,
        }
        dev_metrics = {
            "total_loss": 1.02,
            "classification_loss": 1.0,
            "accuracy": 0.76,
            "macro_f1": 0.71,
            "weighted_f1": 0.73,
        }

        lines = format_epoch_summary(
            epoch=1,
            num_epochs=15,
            epoch_time_seconds=12.34,
            learning_rate=0.0001,
            train_metrics=train_metrics,
            dev_metrics=dev_metrics,
            is_best=True,
            selection_split="dev",
            selection_metric="macro_f1",
            selection_score=0.71,
            best_selection_score=0.71,
            epochs_without_improvement=0,
            patience=3,
        )

        self.assertEqual(
            lines[0],
            "EPOCH 001/015 | time=12.34s | learning_rate=0.000100",
        )
        self.assertEqual(
            lines[1],
            "TRAIN MAIN | total_loss=1.234568 | classification_loss=1.100000 | "
            "accuracy=0.750000 | macro_f1=0.700000 | weighted_f1=0.720000",
        )
        self.assertEqual(
            lines[-2],
            "DEV MAIN | total_loss=1.020000 | classification_loss=1.000000 | "
            "accuracy=0.760000 | macro_f1=0.710000 | weighted_f1=0.730000",
        )
        self.assertEqual(
            lines[-1],
            "STATUS | improved=true | dev_macro_f1=0.710000 | "
            "best_dev_macro_f1=0.710000 | patience=0/3",
        )
        self.assertTrue(any(line.startswith("TRAIN AUX |") for line in lines))
        self.assertTrue(any(line.startswith("TRAIN MODE |") for line in lines))
        self.assertTrue(any(line.startswith("TRAIN FUSION |") for line in lines))
        self.assertTrue(any(line.startswith("TRAIN ROUTING |") for line in lines))
        self.assertIn("TT=[0:0.600000,1:0.400000]", "\n".join(lines))

    def test_split_formatter_only_emits_available_capabilities(self) -> None:
        main = {
            "total_loss": 1.0,
            "classification_loss": 1.0,
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "weighted_f1": 0.45,
        }
        self.assertEqual(len(format_split_metrics("train", main)), 1)

        attention = {**main, "attention_TT_mean": 0.25}
        attention_lines = format_split_metrics("train", attention)
        self.assertEqual([line.split(" |", 1)[0] for line in attention_lines], [
            "TRAIN MAIN",
            "TRAIN FUSION",
        ])

        full = {
            **attention,
            "relation_contrastive_loss": 0.2,
            "mode_auxiliary_loss": 0.3,
            "invariance_loss": 0.1,
            "accuracy_TT": 0.5,
            "reliability_TT_mean": 0.8,
        }
        full_groups = [line.split(" |", 1)[0] for line in format_split_metrics("dev", full)]
        self.assertEqual(
            full_groups,
            ["DEV MAIN", "DEV AUX", "DEV MODE", "DEV FUSION"],
        )

        moe = {
            **full,
            "router_entropy_TT": 1.2,
            "expert_usage_TT_expert_0": 1.0,
        }
        moe_groups = [line.split(" |", 1)[0] for line in format_split_metrics("test", moe)]
        self.assertEqual(
            moe_groups,
            [
                "TEST MAIN",
                "TEST AUX",
                "TEST MODE",
                "TEST FUSION",
                "TEST ROUTING",
                "TEST EXPERTS",
            ],
        )

    def test_jsonl_append_preserves_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "metrics_history.jsonl"
            append_jsonl(history_path, {"run_id": "first", "epoch": 1})
            append_jsonl(history_path, {"run_id": "second", "epoch": 1})

            records = _read_jsonl(history_path)

        self.assertEqual(records, [
            {"run_id": "first", "epoch": 1},
            {"run_id": "second", "epoch": 1},
        ])


class TestTrainerLogging(unittest.TestCase):
    def test_two_epochs_and_consecutive_runs_append_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.pt"
            output_dir = root / "output"
            torch.save(_feature_payload(), feature_path)

            first_trainer = _build_trainer(feature_path, output_dir, num_epochs=2)
            first_run_id = first_trainer.run_id
            try:
                first_trainer.train()
                test_report = first_trainer.evaluate_test(first_trainer.train_loader)
            finally:
                first_trainer.close()

            history_path = output_dir / "metrics_history.jsonl"
            first_records = _read_jsonl(history_path)
            log_text = (output_dir / "train.log").read_text(encoding="utf-8")

            self.assertEqual(len(first_records), 2)
            self.assertEqual([record["epoch"] for record in first_records], [1, 2])
            self.assertTrue(all(record["run_id"] == first_run_id for record in first_records))
            self.assertTrue(all("macro_f1" in record["train"] for record in first_records))
            self.assertTrue(all("weighted_f1" in record["dev"] for record in first_records))
            self.assertTrue(
                all(record["selection_metric"] == "macro_f1" for record in first_records)
            )
            self.assertTrue(all("selection_score" in record for record in first_records))
            self.assertTrue(
                all("best_selection_score" in record for record in first_records)
            )
            self.assertIn(f"RUN START | run_id={first_run_id}", log_text)
            self.assertIn("selection_metric=macro_f1 | selection_mode=max", log_text)
            self.assertIn("EPOCH 001/002", log_text)
            self.assertIn("EPOCH 002/002", log_text)
            self.assertIn("TRAIN MAIN |", log_text)
            self.assertIn("DEV MAIN |", log_text)
            self.assertIn("TEST MAIN |", log_text)
            self.assertIn("STATUS |", log_text)
            self.assertNotIn("TRAIN ROUTING |", log_text)
            self.assertNotIn("TRAIN AUX |", log_text)
            self.assertEqual(test_report["num_samples"], 3)
            self.assertEqual(test_report["selection_metric"], "macro_f1")
            self.assertEqual(test_report["selection_mode"], "max")
            self.assertIn("best_selection_score", test_report)
            self.assertNotIn("best_dev_loss", test_report)
            self.assertTrue((output_dir / "test_metrics.json").is_file())

            second_trainer = _build_trainer(feature_path, output_dir, num_epochs=1)
            second_run_id = second_trainer.run_id
            try:
                second_trainer.train()
            finally:
                second_trainer.close()

            all_records = _read_jsonl(history_path)
            self.assertEqual(len(all_records), 3)
            self.assertNotEqual(first_run_id, second_run_id)
            self.assertEqual(all_records[-1]["run_id"], second_run_id)

    def test_higher_macro_f1_wins_even_when_dev_loss_is_worse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.pt"
            output_dir = root / "output"
            torch.save(_feature_payload(), feature_path)
            trainer = _build_trainer(feature_path, output_dir, num_epochs=3)
            trainer._run_epoch = Mock(
                side_effect=[
                    _scripted_metrics(total_loss=1.0, macro_f1=0.40),
                    _scripted_metrics(total_loss=0.80, macro_f1=0.50),
                    _scripted_metrics(total_loss=0.9, macro_f1=0.45),
                    _scripted_metrics(total_loss=1.20, macro_f1=0.60),
                    _scripted_metrics(total_loss=0.8, macro_f1=0.50),
                    _scripted_metrics(total_loss=0.70, macro_f1=0.55),
                ]
            )
            try:
                trainer.train()
            finally:
                trainer.close()

            checkpoint = torch.load(
                output_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            records = _read_jsonl(output_dir / "metrics_history.jsonl")

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertAlmostEqual(checkpoint["score"], 0.60)
        self.assertEqual(checkpoint["selection_metric"], "macro_f1")
        self.assertEqual(checkpoint["selection_mode"], "max")
        self.assertEqual([record["is_best"] for record in records], [True, True, False])
        self.assertEqual(records[-1]["epochs_without_improvement"], 1)

    def test_equal_macro_f1_advances_patience_and_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.pt"
            output_dir = root / "output"
            torch.save(_feature_payload(), feature_path)
            trainer = _build_trainer(
                feature_path,
                output_dir,
                num_epochs=4,
                patience=1,
            )
            trainer._run_epoch = Mock(
                side_effect=[
                    _scripted_metrics(total_loss=1.0, macro_f1=0.40),
                    _scripted_metrics(total_loss=0.90, macro_f1=0.50),
                    _scripted_metrics(total_loss=0.8, macro_f1=0.45),
                    _scripted_metrics(total_loss=0.70, macro_f1=0.50),
                ]
            )
            try:
                trainer.train()
            finally:
                trainer.close()

            checkpoint = torch.load(
                output_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            records = _read_jsonl(output_dir / "metrics_history.jsonl")
            log_text = (output_dir / "train.log").read_text(encoding="utf-8")

        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(len(records), 2)
        self.assertFalse(records[-1]["is_best"])
        self.assertEqual(records[-1]["epochs_without_improvement"], 1)
        self.assertIn("EARLY STOPPING | epoch=2", log_text)

    def test_legacy_checkpoint_selection_is_recognized_as_total_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.pt"
            output_dir = root / "output"
            torch.save(_feature_payload(), feature_path)
            trainer = _build_trainer(feature_path, output_dir, num_epochs=1)
            try:
                trainer.train()
                checkpoint_path = output_dir / "best_model.pt"
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                checkpoint.pop("selection_metric")
                checkpoint.pop("selection_mode")
                torch.save(checkpoint, checkpoint_path)

                report = trainer.evaluate_test(trainer.train_loader)
            finally:
                trainer.close()

        self.assertEqual(report["selection_metric"], "total_loss")
        self.assertEqual(report["selection_mode"], "min")
        self.assertEqual(report["best_selection_score"], checkpoint["score"])


def _build_trainer(
    feature_path: Path,
    output_dir: Path,
    *,
    num_epochs: int,
    patience: int = 3,
) -> Trainer:
    config = {
        "model_name": "dense_ffn_mean",
        "feature_source": "synthetic",
        "input_dim": 16,
        "num_labels": 3,
        "batch_size": 3,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "aux_loss_coef": 0.0,
        "use_relation_contrastive_loss": False,
        "relation_contrastive_margin": 0.5,
        "relation_contrastive_loss_coef": 0.0,
        "use_mode_auxiliary_loss": False,
        "mode_auxiliary_loss_coef": 0.0,
        "use_invariance_loss": False,
        "invariance_loss_coef": 0.0,
        "reliability_weighted_invariance": False,
        "num_epochs": num_epochs,
        "patience": patience,
    }
    loader = build_feature_loader(feature_path, config, shuffle=False)
    model = DenseFFNMeanNLIModel(
        input_dim=16,
        num_modes=4,
        hidden_dim=8,
        expert_ffn_dim=16,
        num_labels=3,
        dropout=0.0,
    )
    return Trainer(
        model=model,
        train_loader=loader,
        dev_loader=loader,
        config=config,
        output_dir=output_dir,
    )


def _scripted_metrics(*, total_loss: float, macro_f1: float) -> dict[str, float]:
    return {
        "total_loss": total_loss,
        "classification_loss": total_loss,
        "accuracy": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
    }


def _feature_payload() -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for sample_index, label in enumerate(("entailment", "neutral", "contradiction")):
        payload.append(
            {
                "id": f"sample-{sample_index}",
                "sample_index": sample_index,
                "label": label,
                "features": {
                    mode: torch.randn(16)
                    for mode in INPUT_MODE_NAMES
                },
            }
        )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
