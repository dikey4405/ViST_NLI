from __future__ import annotations

import unittest

import torch

from KLTN.source.data.schemas import INPUT_MODE_NAMES, InputMode
from KLTN.source.feature_pipelines.sonar_feature_pipeline.feature_extractor import (
    build_pair_feature,
)
from KLTN.source.feature_pipelines.sonar_feature_pipeline.save_features import (
    extract_batch_features,
)


class FakeSonarEncoder:
    def __init__(self) -> None:
        self.text_calls = 0
        self.speech_calls = 0

    def encode_text(self, values: list[str]) -> torch.Tensor:
        self.text_calls += 1
        return torch.ones(len(values), 1024)

    def encode_speech(self, values: list[str]) -> torch.Tensor:
        self.speech_calls += 1
        return torch.full((len(values), 1024), 2.0)


class TestPairFeature(unittest.TestCase):
    def test_build_pair_feature_uses_absolute_difference(self) -> None:
        u = torch.tensor([1.0, 3.0]).repeat(512)
        v = torch.tensor([2.0, 1.0]).repeat(512)

        feature = build_pair_feature(u, v)

        self.assertEqual(feature.shape, (4096,))
        self.assertTrue(torch.equal(feature[:1024], u))
        self.assertTrue(torch.equal(feature[1024:2048], v))
        self.assertTrue(torch.equal(feature[2048:3072], torch.abs(u - v)))
        self.assertTrue(torch.equal(feature[3072:], u * v))


class TestBatchFeatureExtraction(unittest.TestCase):
    def test_reuses_embeddings_and_keeps_four_separate_modes(self) -> None:
        batch = {"input_pairs": _build_input_pairs(batch_size=2)}
        encoder = FakeSonarEncoder()

        features = extract_batch_features(batch, encoder)

        self.assertEqual(tuple(features), INPUT_MODE_NAMES)
        self.assertEqual(encoder.text_calls, 2)
        self.assertEqual(encoder.speech_calls, 2)
        for mode in INPUT_MODE_NAMES:
            self.assertEqual(features[mode].shape, (2, 4096))


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


if __name__ == "__main__":
    unittest.main()
