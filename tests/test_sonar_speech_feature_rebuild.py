from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from KLTN.source.data.schemas import INPUT_MODE_NAMES, InputMode
from KLTN.source.feature_pipelines.sonar_16k_pipeline.rebuild_speech_features import (
    prepare_split,
    rebuild_split,
    run_rebuild,
    save_features_atomic,
)
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.feature_dataset import FeatureTensorDataset


class FakeSpeechPipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, paths: list[str], batch_size: int) -> torch.Tensor:
        self.calls.extend(paths)
        return torch.stack(
            [torch.full((1024,), 2.0 if "premise_audio" in path else 3.0) for path in paths]
        )


class SonarSpeechFeatureRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw_root = self.root / "Dataset"
        self.source_audio_root = self.raw_root
        self.output_audio_root = self.root / "Dataset_resampled_16k"
        self.old_feature_root = self.root / "Features" / "sonar"
        self.new_feature_root = self.root / "Features" / "sonar_16k"
        self.raw_root.mkdir()
        self.old_feature_root.mkdir(parents=True)

        records = []
        self.old_rows = []
        for index, sample_id in enumerate(("a", "b")):
            records.append(
                {
                    "id": sample_id,
                    "premise": f"premise {sample_id}",
                    "hypothesis": f"hypothesis {sample_id}",
                    "label": "entailment",
                }
            )
            for side, prefix in (("premise", "prem"), ("hypothesis", "hypo")):
                name = f"{prefix}_{sample_id}_entailment.wav"
                source = self.raw_root / "train_audio" / f"{side}_audio" / name
                output = self.output_audio_root / "train_audio" / f"{side}_audio" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(source, np.zeros(480), 48_000)
                sf.write(output, np.zeros(160), 16_000)
            text_premise = torch.full((1024,), float(index + 1))
            text_hypothesis = torch.full((1024,), float(index + 4))
            text_text = build_pair_feature(text_premise, text_hypothesis)
            self.old_rows.append(
                {
                    "id": sample_id,
                    "label": "entailment",
                    "sample_index": index,
                    "features": {
                        mode: text_text if mode == InputMode.TEXT_TEXT.value else torch.zeros(4096)
                        for mode in INPUT_MODE_NAMES
                    },
                }
            )
        (self.raw_root / "train.json").write_text(json.dumps(records), encoding="utf-8")
        torch.save(self.old_rows, self.old_feature_root / "train.pt")

    def _prepare(self):
        return prepare_split(
            self.raw_root,
            self.source_audio_root,
            self.output_audio_root,
            self.old_feature_root,
            self.new_feature_root,
            "train",
        )

    def test_tt_bitwise_preserved_and_only_new_speech_used(self) -> None:
        fake = FakeSpeechPipeline()
        count = rebuild_split(
            self.raw_root,
            self.source_audio_root,
            self.output_audio_root,
            self.old_feature_root,
            self.new_feature_root,
            "train",
            speech_pipeline=fake,
            speech_batch_size=2,
        )
        self.assertEqual(count, 2)
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(len(set(fake.calls)), 4)
        self.assertTrue(all("Dataset_resampled_16k" in path for path in fake.calls))

        result = torch.load(self.new_feature_root / "train.pt", map_location="cpu", weights_only=False)
        for index, row in enumerate(result):
            old_tt = self.old_rows[index]["features"][InputMode.TEXT_TEXT.value]
            u, v = old_tt[:1024], old_tt[1024:2048]
            speech_u = torch.full((1024,), 2.0)
            speech_v = torch.full((1024,), 3.0)
            self.assertEqual(tuple(row["features"]), INPUT_MODE_NAMES)
            self.assertEqual(row["sample_index"], index)
            self.assertEqual(row["id"], self.old_rows[index]["id"])
            self.assertEqual(row["label"], "entailment")
            self.assertTrue(torch.equal(row["features"]["text_text"], old_tt))
            self.assertTrue(
                torch.equal(row["features"]["text_speech"], build_pair_feature(u, speech_v))
            )
            self.assertTrue(
                torch.equal(row["features"]["speech_text"], build_pair_feature(speech_u, v))
            )
            self.assertTrue(
                torch.equal(
                    row["features"]["speech_speech"],
                    build_pair_feature(speech_u, speech_v),
                )
            )
            for feature in row["features"].values():
                self.assertEqual(feature.shape, (4096,))

        loaded = FeatureTensorDataset(self.new_feature_root / "train.pt")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["features"].shape, (4, 4096))

    def test_alignment_mismatch_stops_before_loading_model(self) -> None:
        del self.old_rows[0]["sample_index"]
        torch.save(self.old_rows, self.old_feature_root / "train.pt")
        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.rebuild_speech_features.load_speech_pipeline"
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "sample_index"):
                run_rebuild(
                    self.raw_root,
                    self.source_audio_root,
                    self.output_audio_root,
                    self.old_feature_root,
                    self.new_feature_root,
                    ["train"],
                )
            load_model.assert_not_called()

    def test_missing_or_invalid_resampled_audio_fails_preflight(self) -> None:
        target = next((self.output_audio_root / "train_audio" / "premise_audio").glob("*.wav"))
        target.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing resampled WAV"):
            self._prepare()

        sf.write(target, np.zeros(160), 48_000)
        with self.assertRaisesRegex(ValueError, "mono 16 kHz"):
            self._prepare()

    def test_output_exists_by_default_and_failed_save_is_not_published(self) -> None:
        output_path = self.new_feature_root / "train.pt"
        self.new_feature_root.mkdir(parents=True)
        torch.save([], output_path)
        with self.assertRaises(FileExistsError):
            self._prepare()
        output_path.unlink()

        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.rebuild_speech_features.torch.save",
            side_effect=OSError("interrupted"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                save_features_atomic(self.old_rows, output_path)
        self.assertFalse(output_path.exists())
        self.assertEqual(list(self.new_feature_root.iterdir()), [])

    def test_run_rebuild_does_not_initialize_text_encoder(self) -> None:
        fake = FakeSpeechPipeline()
        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.rebuild_speech_features.load_speech_pipeline",
            return_value=fake,
        ), patch(
            "KLTN.source.feature_pipelines.sonar_feature_pipeline.sonar_encoder.SonarEncoder.__init__",
            side_effect=AssertionError("text encoder must not be loaded"),
        ):
            run_rebuild(
                self.raw_root,
                self.source_audio_root,
                self.output_audio_root,
                self.old_feature_root,
                self.new_feature_root,
                ["train"],
            )
        self.assertEqual(len(fake.calls), 4)


if __name__ == "__main__":
    unittest.main()
