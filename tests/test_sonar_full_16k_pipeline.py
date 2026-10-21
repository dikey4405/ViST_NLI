from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from KLTN.source.data.schemas import INPUT_MODE_NAMES
from KLTN.source.feature_pipelines.sonar_16k_pipeline.config import FullSonar16kConfig
from KLTN.source.feature_pipelines.sonar_16k_pipeline.dataset import FullSonar16kDataset
from KLTN.source.feature_pipelines.sonar_16k_pipeline.extract_features import (
    run_extraction,
    save_features_atomic,
)
from KLTN.source.feature_pipelines.sonar_16k_pipeline.sonar_encoder import FullSonarEncoder
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.feature_dataset import FeatureTensorDataset


class FakeEncoder:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.speech_calls: list[list[str]] = []

    @staticmethod
    def text_value(text: str) -> float:
        return float(sum(ord(char) for char in text) % 17 + 1)

    @staticmethod
    def speech_value(path: str) -> float:
        return 21.0 if "premise_audio" in path else 31.0

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        self.text_calls.append(list(texts))
        return torch.stack([torch.full((1024,), self.text_value(text)) for text in texts])

    def encode_speech(self, paths: list[str]) -> torch.Tensor:
        self.speech_calls.append(list(paths))
        return torch.stack([torch.full((1024,), self.speech_value(path)) for path in paths])


class FullSonar16kPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw_root = self.root / "Dataset"
        self.audio_root = self.root / "Dataset_resampled_16k"
        self.output_root = self.root / "Features" / "sonar_full_16k"
        self.raw_root.mkdir()

    def _record(self, sample_id: str, label: str = "entailment") -> dict[str, str]:
        return {
            "id": sample_id,
            "premise": f"premise {sample_id}",
            "hypothesis": f"hypothesis {sample_id}",
            "gold_label": label,
        }

    def _wav(self, split: str, side: str, name: str, rate: int = 16_000) -> Path:
        path = self.audio_root / f"{split}_audio" / f"{side}_audio" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(rate // 100, dtype=np.float32), rate)
        return path

    def _split(self, split: str, records: list[dict[str, str]]) -> None:
        (self.raw_root / f"{split}.json").write_text(json.dumps(records), encoding="utf-8")
        for record in records:
            sample_id = record["id"]
            label = record["gold_label"]
            self._wav(split, "premise", f"prem_{sample_id}_{label}.wav")
            self._wav(split, "hypothesis", f"hypo_{sample_id}_{label}.wav")

    def _config(self, *splits: str, batch_size: int = 8) -> FullSonar16kConfig:
        return FullSonar16kConfig(
            raw_data_dir=self.raw_root,
            audio_root_16k=self.audio_root,
            output_dir=self.output_root,
            splits=splits,
            batch_size=batch_size,
        )

    def test_full_extraction_uses_fresh_text_and_speech_and_preserves_schema(self) -> None:
        records = [self._record("a"), self._record("b", "contradiction")]
        self._split("train", records)
        fake = FakeEncoder()

        result_paths = run_extraction(self._config("train"), encoder=fake)

        self.assertEqual(set(result_paths), {"train"})
        self.assertEqual(len(fake.text_calls), 1)
        self.assertEqual(len(fake.speech_calls), 1)
        self.assertEqual(set(fake.text_calls[0]), {
            "premise a", "hypothesis a", "premise b", "hypothesis b"
        })
        self.assertEqual(len(fake.speech_calls[0]), 4)
        self.assertTrue(
            all(
                Path(path).is_relative_to(self.audio_root.resolve())
                for path in fake.speech_calls[0]
            )
        )
        self.assertFalse((self.root / "Features" / "sonar").exists())
        self.assertFalse((self.root / "Features" / "sonar_16k").exists())

        payload = torch.load(result_paths["train"], map_location="cpu", weights_only=False)
        self.assertEqual(len(payload), len(records))
        for index, row in enumerate(payload):
            self.assertEqual(row["id"], records[index]["id"])
            self.assertEqual(row["label"], records[index]["gold_label"])
            self.assertEqual(row["sample_index"], index)
            self.assertEqual(tuple(row["features"]), INPUT_MODE_NAMES)
            u = torch.full((1024,), fake.text_value(records[index]["premise"]))
            v = torch.full((1024,), fake.text_value(records[index]["hypothesis"]))
            speech_u = torch.full((1024,), 21.0)
            speech_v = torch.full((1024,), 31.0)
            expected = (
                build_pair_feature(u, v),
                build_pair_feature(u, speech_v),
                build_pair_feature(speech_u, v),
                build_pair_feature(speech_u, speech_v),
            )
            for mode, expected_feature in zip(INPUT_MODE_NAMES, expected):
                self.assertTrue(torch.equal(row["features"][mode], expected_feature))
                self.assertEqual(row["features"][mode].shape, (4096,))

        loaded = FeatureTensorDataset(result_paths["train"])
        self.assertEqual(len(loaded), len(records))
        self.assertEqual(loaded[0]["features"].shape, (4, 4096))

    def test_encoder_wrapper_loads_both_pretrained_pipelines_once(self) -> None:
        class FakeTextPipeline:
            def predict(self, texts, source_lang, batch_size):
                if source_lang != "vie_Latn" or batch_size != 8:
                    raise AssertionError("Unexpected text encoder settings")
                return torch.ones(len(texts), 1024)

        class FakeSpeechPipeline:
            def predict(self, paths, batch_size):
                if batch_size != 8:
                    raise AssertionError("Unexpected speech encoder batch size")
                return torch.full((len(paths), 1024), 2.0)

        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.sonar_encoder.load_text_pipeline",
            return_value=FakeTextPipeline(),
        ) as load_text, patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.sonar_encoder.load_speech_pipeline",
            return_value=FakeSpeechPipeline(),
        ) as load_speech:
            encoder = FullSonarEncoder(self._config("train"))
            self.assertEqual(encoder.encode_text(["mot cau"]).shape, (1, 1024))
            self.assertEqual(encoder.encode_speech(["audio.wav"]).shape, (1, 1024))
        load_text.assert_called_once()
        load_speech.assert_called_once()

    def test_explicit_windows_and_posix_paths_use_only_16k_output(self) -> None:
        record = self._record("a")
        record["premise_audio"] = r"D:\old\train_audio\premise_audio\prem_a_entailment.wav"
        record["hypothesis_audio"] = (
            "/unavailable/train_audio/hypothesis_audio/hypo_a_entailment.wav"
        )
        self._split("train", [record])
        dataset = FullSonar16kDataset(
            self.raw_root / "train.json", self.audio_root, "train"
        )
        pair = dataset[0]["input_pairs"]["speech_speech"]
        self.assertTrue(Path(pair["premise"]).is_relative_to(self.audio_root.resolve()))
        self.assertTrue(Path(pair["hypothesis"]).is_relative_to(self.audio_root.resolve()))

    def test_wrong_rate_and_missing_16k_wav_fail_before_encoder_load(self) -> None:
        self._split("train", [self._record("a")])
        premise = self.audio_root / "train_audio" / "premise_audio" / "prem_a_entailment.wav"
        self._wav("train", "premise", premise.name, rate=48_000)
        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.extract_features.FullSonarEncoder"
        ) as load_encoder:
            with self.assertRaisesRegex(ValueError, "sample_rate=48000"):
                run_extraction(self._config("train"))
            load_encoder.assert_not_called()

        premise.unlink()
        raw_premise = self.raw_root / "train_audio" / "premise_audio" / premise.name
        raw_premise.parent.mkdir(parents=True)
        sf.write(raw_premise, np.zeros(480, dtype=np.float32), 48_000)
        with self.assertRaisesRegex(FileNotFoundError, "sample_index=0.*side=premise"):
            run_extraction(self._config("train"), encoder=FakeEncoder())

    def test_reuses_duplicate_text_embeddings_within_batch(self) -> None:
        records = [self._record("a"), self._record("b")]
        records[1]["premise"] = records[0]["premise"]
        self._split("train", records)
        fake = FakeEncoder()

        run_extraction(self._config("train"), encoder=fake)

        self.assertEqual(len(fake.text_calls), 1)
        self.assertEqual(len(fake.text_calls[0]), 3)
        self.assertEqual(len(fake.speech_calls), 1)
        self.assertEqual(len(fake.speech_calls[0]), 4)

    def test_distinct_explicit_sources_with_one_destination_are_rejected(self) -> None:
        records = [self._record("a"), self._record("b")]
        records[0]["premise_audio"] = r"C:\source_one\same.wav"
        records[1]["premise_audio"] = r"D:\source_two\same.wav"
        self._split("train", records)
        self._wav("train", "premise", "same.wav")
        with self.assertRaisesRegex(ValueError, "distinct explicit audio references"):
            FullSonar16kDataset(self.raw_root / "train.json", self.audio_root, "train")

    def test_duplicate_id_needs_unambiguous_audio_mapping(self) -> None:
        records = [self._record("a"), self._record("a")]
        records[1]["premise"] = "a different premise"
        self._split("train", records)
        with self.assertRaisesRegex(ValueError, "duplicate ID/audio mapping is ambiguous"):
            FullSonar16kDataset(self.raw_root / "train.json", self.audio_root, "train")

        records[0]["premise_audio"] = "first.wav"
        records[1]["premise_audio"] = "second.wav"
        (self.raw_root / "train.json").write_text(json.dumps(records), encoding="utf-8")
        self._wav("train", "premise", "first.wav")
        self._wav("train", "premise", "second.wav")
        dataset = FullSonar16kDataset(self.raw_root / "train.json", self.audio_root, "train")
        self.assertEqual(len(dataset), 2)
        self.assertEqual([row["sample_index"] for row in dataset], [0, 1])

    def test_selected_splits_only_and_existing_output_is_not_replaced(self) -> None:
        self._split("dev", [self._record("dev")])
        self._split("test", [self._record("test")])
        result = run_extraction(self._config("dev", "test"), encoder=FakeEncoder())
        self.assertEqual(set(result), {"dev", "test"})
        self.assertFalse((self.output_root / "train.pt").exists())
        original_bytes = result["dev"].read_bytes()
        with self.assertRaises(FileExistsError):
            run_extraction(self._config("dev"), encoder=FakeEncoder())
        self.assertEqual(result["dev"].read_bytes(), original_bytes)

    def test_failed_atomic_save_leaves_no_published_feature(self) -> None:
        output = self.output_root / "train.pt"
        with patch(
            "KLTN.source.feature_pipelines.sonar_16k_pipeline.extract_features.torch.save",
            side_effect=OSError("interrupted"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                save_features_atomic([], output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_output_cannot_overlap_old_feature_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "Output directory must be separate"):
            FullSonar16kConfig(output_dir=Path("Features/sonar"))


if __name__ == "__main__":
    unittest.main()
