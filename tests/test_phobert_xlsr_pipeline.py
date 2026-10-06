from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader

from KLTN.source.data.config import DEFAULT_LABEL_MAPPING
from KLTN.source.data.schemas import INPUT_MODE_NAMES
from KLTN.source.feature_pipelines import phobert_xlsr_pipeline as pipeline_package
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.alignment_loss import (
    SymmetricTextSpeechAlignmentLoss,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.config import (
    PhoBERTXLSRConfig,
    load_pipeline_config,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.cache_embeddings import (
    cache_split_embeddings,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.cache_store import (
    modality_cache_directory,
    save_manifest,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.cache_identity import (
    compute_cache_fingerprint,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.extract_features import (
    extract_split_features,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.pair_feature_builder import (
    build_four_mode_features,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.pooling import masked_mean_pooling
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.projectors import (
    SpeechProjector,
    TextProjector,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.schemas import (
    AudioOverlengthPolicy,
    BEST_PROJECTOR_CHECKPOINT_NAME,
    CacheManifest,
    CacheModality,
    MODE_ORDER,
    PAIR_FEATURE_COMPONENTS,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.speech_encoder import (
    AudioOverlengthError,
    AudioSkippedError,
    XLSRSpeechEncoder,
    load_audio,
)
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.train_alignment import train_alignment
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.utils import freeze_module
from KLTN.source.feature_pipelines.phobert_xlsr_pipeline.validate_features import (
    validate_feature_outputs,
)
from KLTN.source.pair_features import build_pair_feature
from KLTN.source.training.feature_dataset import FeatureBatchCollator, FeatureTensorDataset
from KLTN.source.training.trainer import resolve_feature_source_paths


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "phobert_xlsr_pipeline.yaml"


class PoolingTests(unittest.TestCase):
    def test_masked_mean_pooling_ignores_padding(self) -> None:
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean_pooling(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_text_and_speech_pooling_shapes(self) -> None:
        text = masked_mean_pooling(torch.randn(2, 7, 1024), torch.ones(2, 7))
        speech = masked_mean_pooling(torch.randn(3, 11, 1024), torch.ones(3, 11))
        self.assertEqual(text.shape, (2, 1024))
        self.assertEqual(speech.shape, (3, 1024))

    def test_pooling_rejects_mismatched_mask(self) -> None:
        with self.assertRaises(ValueError):
            masked_mean_pooling(torch.randn(2, 4, 8), torch.ones(2, 3))


class ProjectorAndAlignmentTests(unittest.TestCase):
    def test_projector_shapes_and_independent_parameters(self) -> None:
        text_projector = TextProjector(1024, 64, 1024, 0.0)
        speech_projector = SpeechProjector(1024, 64, 1024, 0.0)
        self.assertEqual(text_projector(torch.randn(2, 1024)).shape, (2, 1024))
        self.assertEqual(speech_projector(torch.randn(2, 1024)).shape, (2, 1024))
        text_parameter_ids = {id(parameter) for parameter in text_projector.parameters()}
        speech_parameter_ids = {id(parameter) for parameter in speech_projector.parameters()}
        self.assertFalse(text_parameter_ids & speech_parameter_ids)

    def test_alignment_loss_is_finite_scalar_and_backpropagates(self) -> None:
        text_projector = TextProjector(1024, 32, 1024, 0.0)
        speech_projector = SpeechProjector(1024, 32, 1024, 0.0)
        criterion = SymmetricTextSpeechAlignmentLoss(temperature=0.07)
        text = text_projector(torch.randn(4, 1024))
        speech = speech_projector(torch.randn(4, 1024))
        loss = criterion(text, speech)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in text_projector.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in speech_projector.parameters())
        )

    def test_alignment_batch_size_one_is_finite(self) -> None:
        text = torch.randn(1, 1024, requires_grad=True)
        speech = torch.randn(1, 1024, requires_grad=True)
        loss = SymmetricTextSpeechAlignmentLoss()(text, speech)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(speech.grad)

    def test_frozen_encoder_parameters_receive_no_gradient(self) -> None:
        encoder = nn.Linear(8, 8)
        freeze_module(encoder)
        inputs = torch.randn(2, 8, requires_grad=True)
        encoder(inputs).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))


class PairFeatureTests(unittest.TestCase):
    def test_pair_feature_shape_and_formula(self) -> None:
        u = torch.randn(3, 1024)
        v = torch.randn(3, 1024)
        feature = build_pair_feature(u, v)
        self.assertEqual(feature.shape, (3, 4096))
        torch.testing.assert_close(feature[:, 2048:3072], torch.abs(u - v))
        torch.testing.assert_close(feature[:, 3072:], u * v)

    def test_four_modes_have_fixed_order_and_shape(self) -> None:
        embeddings = [torch.randn(2, 1024) for _ in range(4)]
        features = build_four_mode_features(*embeddings)
        self.assertEqual(MODE_ORDER, INPUT_MODE_NAMES)
        self.assertEqual(features.shape, (2, 4, 4096))
        expected_pairs = (
            (embeddings[0], embeddings[2]),
            (embeddings[0], embeddings[3]),
            (embeddings[1], embeddings[2]),
            (embeddings[1], embeddings[3]),
        )
        for mode_index, (u, v) in enumerate(expected_pairs):
            torch.testing.assert_close(features[:, mode_index], build_pair_feature(u, v))
        self.assertTrue(torch.isfinite(features).all())

    def test_wrong_embedding_dimension_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_four_mode_features(*[torch.randn(2, 16) for _ in range(4)])

    def test_missing_mode_raises_in_feature_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.pt"
            torch.save(
                [
                    {
                        "id": "sample",
                        "sample_index": 0,
                        "label": "entailment",
                        "features": {"text_text": torch.zeros(4096)},
                    }
                ],
                path,
            )
            with self.assertRaises(ValueError):
                FeatureTensorDataset(path)


class AudioTests(unittest.TestCase):
    def test_stereo_audio_is_mono_and_resampled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stereo_8khz.wav"
            time_axis = np.arange(800, dtype=np.float32) / 8000.0
            stereo = np.stack(
                [np.sin(2 * np.pi * 220 * time_axis), np.sin(2 * np.pi * 440 * time_axis)],
                axis=1,
            )
            sf.write(path, stereo, 8000)
            loaded = load_audio(
                path,
                target_sample_rate=16000,
                max_audio_seconds=1.0,
                overlength_policy=AudioOverlengthPolicy.TRUNCATE,
            )
            self.assertEqual(loaded.waveform.ndim, 1)
            self.assertEqual(loaded.original_num_channels, 2)
            self.assertEqual(loaded.sample_rate, 16000)
            self.assertTrue(loaded.was_resampled)
            self.assertEqual(loaded.waveform.shape[0], 1600)

    def test_audio_duration_policies_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "long.wav"
            sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
            truncated = load_audio(
                path,
                target_sample_rate=16000,
                max_audio_seconds=0.05,
                overlength_policy=AudioOverlengthPolicy.TRUNCATE,
            )
            self.assertTrue(truncated.was_truncated)
            self.assertEqual(truncated.waveform.shape[0], 800)
            with self.assertRaises(AudioSkippedError):
                load_audio(
                    path,
                    target_sample_rate=16000,
                    max_audio_seconds=0.05,
                    overlength_policy=AudioOverlengthPolicy.SKIP,
                )
            with self.assertRaises(AudioOverlengthError):
                load_audio(
                    path,
                    target_sample_rate=16000,
                    max_audio_seconds=0.05,
                    overlength_policy=AudioOverlengthPolicy.RAISE,
                )

    def test_waveform_mask_is_downsampled_to_feature_length(self) -> None:
        class FakeXLSRModel(nn.Module):
            @staticmethod
            def _get_feat_extract_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
                return lengths // 2

        encoder = XLSRSpeechEncoder.__new__(XLSRSpeechEncoder)
        nn.Module.__init__(encoder)
        encoder.model = FakeXLSRModel()
        waveform_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
        )
        feature_mask = encoder._build_feature_attention_mask(4, waveform_mask)
        expected = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        torch.testing.assert_close(feature_mask, expected)


class PipelineBoundaryTests(unittest.TestCase):
    def test_new_pipeline_does_not_import_sonar_or_moe_models(self) -> None:
        package_directory = Path(pipeline_package.__file__).resolve().parent
        forbidden_modules: list[tuple[str, str]] = []
        for path in package_directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    lowered = module.lower()
                    if "sonar" in lowered or ".models" in lowered:
                        forbidden_modules.append((path.name, module))
        self.assertEqual(forbidden_modules, [])

    def test_feature_source_resolver_selects_only_one_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {
                "feature_source": "phobert_xlsr",
                "feature_sources": {
                    "sonar": {
                        split: root / "sonar" / f"{split}.pt"
                        for split in ("train", "dev", "test")
                    },
                    "phobert_xlsr": {
                        split: root / "new" / f"{split}.pt"
                        for split in ("train", "dev", "test")
                    },
                },
            }
            source, paths = resolve_feature_source_paths(config)
            self.assertEqual(source, "phobert_xlsr")
            self.assertTrue(all(path.parent.name == "new" for path in paths.values()))


class CacheResumeTests(unittest.TestCase):
    def test_text_cache_is_sharded_and_complete_cache_skips_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _cache_test_config(root)
            encoder_instances = 0

            class FakeTextEncoder:
                def __init__(self, **_: object) -> None:
                    nonlocal encoder_instances
                    encoder_instances += 1

                def encode(self, texts: list[str]) -> torch.Tensor:
                    values = torch.tensor(
                        [float(len(text)) for text in texts], dtype=torch.float32
                    )
                    return values.unsqueeze(1).repeat(1, 1024)

            target = (
                "KLTN.source.feature_pipelines.phobert_xlsr_pipeline."
                "cache_embeddings.PhoBERTTextEncoder"
            )
            with patch(target, FakeTextEncoder):
                manifest = cache_split_embeddings(config, "train", CacheModality.TEXT)
                resumed_manifest = cache_split_embeddings(
                    config, "train", CacheModality.TEXT
                )

            self.assertTrue(manifest.complete)
            self.assertEqual(manifest.processed_sample_indices, (0, 1, 2))
            self.assertEqual(len(manifest.shards), 2)
            self.assertEqual(resumed_manifest, manifest)
            self.assertEqual(encoder_instances, 1)


class FakePipelineIntegrationTests(unittest.TestCase):
    def test_fake_cache_to_moe_loader_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _temporary_config(root)
            for split in ("train", "dev", "test"):
                _write_fake_cache(config, split, num_samples=4)

            checkpoint_path = train_alignment(config)
            self.assertEqual(checkpoint_path.name, BEST_PROJECTOR_CHECKPOINT_NAME)
            self.assertTrue(checkpoint_path.exists())
            for split in ("train", "dev", "test"):
                output_path = extract_split_features(config, split)
                dataset = FeatureTensorDataset(output_path)
                loader = DataLoader(dataset, batch_size=4, collate_fn=FeatureBatchCollator())
                batch = next(iter(loader))
                self.assertEqual(batch["features"].shape, (4, 4, 4096))
                self.assertTrue(torch.isfinite(batch["features"]).all())

            report = validate_feature_outputs(config)
            self.assertEqual(report["status"], "passed")
            metadata = json.loads(
                (config.output.directory / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["text_encoder"], "vinai/phobert-large")
            self.assertEqual(
                metadata["speech_encoder"], "facebook/wav2vec2-xls-r-300m"
            )
            self.assertEqual(metadata["pair_feature_components"], list(PAIR_FEATURE_COMPONENTS))
            self.assertEqual(metadata["mode_order"], list(INPUT_MODE_NAMES))


def _temporary_config(root: Path) -> PhoBERTXLSRConfig:
    base = load_pipeline_config(CONFIG_PATH)
    return replace(
        base,
        device="cpu",
        mixed_precision=False,
        data=replace(base.data, raw_data_dir=root / "Dataset", num_workers=0, pin_memory=False),
        projector=replace(base.projector, hidden_dim=16, dropout=0.0),
        alignment=replace(base.alignment, batch_size=4, epochs=2, patience=1),
        cache=replace(base.cache, directory=root / "cache", overwrite=False),
        checkpoint=replace(base.checkpoint, directory=root / "checkpoints"),
        output=replace(
            base.output,
            directory=root / "features",
            overwrite=True,
            dtype="float32",
        ),
    )


def _cache_test_config(root: Path) -> PhoBERTXLSRConfig:
    data_root = root / "Data"
    split_directory = data_root / "Train"
    (split_directory / "Premise").mkdir(parents=True)
    (split_directory / "Hypothesis").mkdir(parents=True)
    records = []
    for index in range(3):
        premise_audio = split_directory / f"premise_{index}.wav"
        hypothesis_audio = split_directory / f"hypothesis_{index}.wav"
        premise_audio.touch()
        hypothesis_audio.touch()
        records.append(
            {
                "id": f"sample-{index}",
                "premise": f"premise {index}",
                "hypothesis": f"hypothesis {index}",
                "premise_audio": premise_audio.name,
                "hypothesis_audio": hypothesis_audio.name,
                "gold_label": "entailment",
            }
        )
    data_path = split_directory / "train.json"
    data_path.write_text(json.dumps(records), encoding="utf-8")

    base = load_pipeline_config(CONFIG_PATH)
    split_files = {
        "train": Path("Train/train.json"),
        "dev": Path("Train/train.json"),
        "test": Path("Train/train.json"),
    }
    return replace(
        base,
        device="cpu",
        mixed_precision=False,
        data=replace(
            base.data,
            raw_data_dir=data_root,
            split_files=split_files,
            num_workers=0,
            pin_memory=False,
        ),
        text_encoder=replace(
            base.text_encoder,
            batch_size=2,
            use_word_segmentation=False,
            cache_segmented_text=False,
        ),
        cache=replace(
            base.cache,
            directory=root / "cache",
            shard_size=2,
            overwrite=False,
        ),
        output=replace(base.output, directory=root / "features"),
    )


def _write_fake_cache(
    config: PhoBERTXLSRConfig,
    split: str,
    *,
    num_samples: int,
) -> None:
    ids = [f"{split}-{index}" for index in range(num_samples)]
    labels = list(DEFAULT_LABEL_MAPPING)[:num_samples]
    labels = [labels[index % len(labels)] for index in range(num_samples)]
    sample_indices = torch.arange(num_samples, dtype=torch.long)
    source = config.data.split_path(split)
    audio_root = source.parent / f"{source.stem}_audio"
    records = []
    for index, (sample_id, label) in enumerate(zip(ids, labels)):
        records.append({
            "id": sample_id,
            "gold_label": label,
            "premise": f"premise {index}",
            "hypothesis": f"hypothesis {index}",
        })
        for side, prefix in (("premise", "prem"), ("hypothesis", "hypo")):
            directory = audio_root / f"{side}_audio"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{prefix}_{sample_id}_{label}.wav").touch()
    source.write_text(json.dumps(records), encoding="utf-8")
    for modality in (CacheModality.TEXT, CacheModality.SPEECH):
        directory = modality_cache_directory(config, split, modality)
        directory.mkdir(parents=True, exist_ok=True)
        shard_name = "shard_00000.pt"
        torch.save(
            {
                "ids": ids,
                "labels": labels,
                "sample_indices": sample_indices,
                "premise_embeddings": torch.randn(num_samples, 1024),
                "hypothesis_embeddings": torch.randn(num_samples, 1024),
            },
            directory / shard_name,
        )
        save_manifest(
            CacheManifest(
                pipeline_name=config.pipeline_name,
                split=split,
                modality=modality.value,
                num_source_samples=num_samples,
                embedding_dim=1024,
                dtype="float32",
                shard_size=config.cache.shard_size,
                shards=(shard_name,),
                processed_sample_indices=tuple(range(num_samples)),
                skipped_sample_indices=(),
                encoder_name=(
                    config.text_encoder.model_name
                    if modality == CacheModality.TEXT
                    else config.speech_encoder.model_name
                ),
                source_data_path=str(config.data.split_path(split).resolve()),
                truncated_audio_count=0,
                error_count=0,
                complete=True,
                cache_fingerprint=compute_cache_fingerprint(config, split, modality),
            ),
            config,
            split,
            modality,
        )

if __name__ == "__main__":
    unittest.main()
