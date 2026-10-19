from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from KLTN.source.feature_pipelines.sonar_feature_pipeline.resample_audio import (
    collect_audio_jobs,
    load_split_dataset,
    resample_wav,
)


class SonarAudioResamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.wav"
        self.destination = self.root / "output" / "result.wav"

    def test_48k_resampling_preserves_pitch_duration_and_suppresses_alias(self) -> None:
        time = np.arange(48_000) / 48_000
        waveform = 0.3 * np.sin(2 * np.pi * 1_000 * time)
        waveform += 0.3 * np.sin(2 * np.pi * 12_000 * time)
        sf.write(self.source, waveform, 48_000, subtype="FLOAT")
        original_bytes = self.source.read_bytes()

        self.assertEqual(resample_wav(self.source, self.destination), "written")

        output, rate = sf.read(self.destination)
        spectrum = np.abs(np.fft.rfft(output)) * 2 / len(output)
        self.assertEqual(rate, 16_000)
        self.assertEqual(len(output), 16_000)
        self.assertAlmostEqual(float(spectrum[1_000]), 0.3, delta=0.01)
        self.assertLess(float(spectrum[4_000]), 0.01)
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_16k_copy_resume_and_overwrite(self) -> None:
        sf.write(self.source, np.linspace(-0.5, 0.5, 320), 16_000, subtype="PCM_16")
        self.assertEqual(resample_wav(self.source, self.destination), "written")
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())
        self.assertEqual(resample_wav(self.source, self.destination, resume=True), "skipped")
        with self.assertRaises(FileExistsError):
            resample_wav(self.source, self.destination)

        sf.write(self.destination, np.zeros(320), 16_000, subtype="PCM_16")
        with self.assertRaisesRegex(ValueError, "byte-for-byte"):
            resample_wav(self.source, self.destination, resume=True)
        self.assertEqual(resample_wav(self.source, self.destination, overwrite=True), "written")
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())

    def test_rejects_other_rate_stereo_empty_and_nonfinite(self) -> None:
        sf.write(self.source, np.zeros(100), 44_100)
        with self.assertRaisesRegex(ValueError, "48000 or 16000"):
            resample_wav(self.source, self.destination)

        sf.write(self.source, np.zeros((100, 2)), 48_000)
        with self.assertRaisesRegex(ValueError, "mono WAV"):
            resample_wav(self.source, self.destination)

        sf.write(self.source, np.empty(0), 48_000)
        with self.assertRaisesRegex(ValueError, "Empty"):
            resample_wav(self.source, self.destination)

        sf.write(self.source, np.array([0.0, np.nan, 0.0]), 48_000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            resample_wav(self.source, self.destination)

    def test_interrupted_write_leaves_no_published_wav(self) -> None:
        sf.write(self.source, np.zeros(480), 48_000)
        original_bytes = self.source.read_bytes()
        with patch(
            "KLTN.source.feature_pipelines.sonar_feature_pipeline.resample_audio.sf.write",
            side_effect=OSError("interrupted"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                resample_wav(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_rejects_identical_source_and_output_path(self) -> None:
        sf.write(self.source, np.zeros(100), 16_000)
        with self.assertRaisesRegex(ValueError, "identical"):
            resample_wav(self.source, self.source)

    def test_same_source_on_both_sides_keeps_two_output_paths(self) -> None:
        raw_root = self.root / "Dataset"
        split_root = raw_root / "train_audio"
        (split_root / "premise_audio").mkdir(parents=True)
        (split_root / "hypothesis_audio").mkdir()
        shared = split_root / "premise_audio" / "prem_a_entailment.wav"
        sf.write(shared, np.zeros(100), 48_000)
        record = {
            "id": "a",
            "premise": "premise a",
            "hypothesis": "hypothesis a",
            "label": "entailment",
            "premise_audio": "premise_audio/prem_a_entailment.wav",
            "hypothesis_audio": "premise_audio/prem_a_entailment.wav",
        }
        (raw_root / "train.json").write_text(json.dumps([record]), encoding="utf-8")
        dataset = load_split_dataset(raw_root, raw_root, "train")
        jobs = collect_audio_jobs(
            dataset, split_root, self.root / "Dataset_resampled_16k" / "train_audio"
        )
        self.assertEqual(len(jobs), 2)
        self.assertNotEqual(jobs[("premise", shared)], jobs[("hypothesis", shared)])

    def test_collision_and_audio_outside_declared_root(self) -> None:
        raw_root = self.root / "Dataset"
        split_root = raw_root / "train_audio"
        (split_root / "premise_audio").mkdir(parents=True)
        (split_root / "hypothesis_audio").mkdir()
        for folder in ("first", "second"):
            (split_root / folder).mkdir()
            sf.write(split_root / folder / "same.wav", np.zeros(100), 48_000)
        for sample_id in ("a", "b"):
            sf.write(
                split_root / "hypothesis_audio" / f"hypo_{sample_id}_entailment.wav",
                np.zeros(100),
                48_000,
            )
        records = [
            {
                "id": sample_id,
                "premise": f"premise {sample_id}",
                "hypothesis": f"hypothesis {sample_id}",
                "label": "entailment",
                "premise_audio": f"{folder}/same.wav",
                "hypothesis_audio": f"hypothesis_audio/hypo_{sample_id}_entailment.wav",
            }
            for sample_id, folder in (("a", "first"), ("b", "second"))
        ]
        (raw_root / "train.json").write_text(json.dumps(records), encoding="utf-8")
        dataset = load_split_dataset(raw_root, raw_root, "train")
        with self.assertRaisesRegex(ValueError, "collision"):
            collect_audio_jobs(dataset, split_root, self.root / "Dataset_resampled_16k/train_audio")

        outside = self.root / "outside.wav"
        sf.write(outside, np.zeros(100), 48_000)
        records[0]["premise_audio"] = str(outside)
        (raw_root / "train.json").write_text(json.dumps(records), encoding="utf-8")
        dataset = load_split_dataset(raw_root, raw_root, "train")
        with self.assertRaisesRegex(ValueError, "outside declared audio root"):
            collect_audio_jobs(dataset, split_root, self.root / "Dataset_resampled_16k/train_audio")


if __name__ == "__main__":
    unittest.main()
