from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ...data.schemas import INPUT_MODE_NAMES, InputMode
from ...pair_features import build_pair_feature, split_pair_feature
from .resample_audio import (
    PROJECT_ROOT,
    SPLITS,
    TARGET_SAMPLE_RATE,
    collect_audio_jobs,
    load_split_dataset,
    publish_temp_file,
)
from ..sonar_feature_pipeline.schemas import NLIFeatureSample
from .sonar_encoder import resolve_device


@dataclass(frozen=True)
class PreparedRow:
    id: str | None
    label: str
    sample_index: int
    text_text: torch.Tensor
    premise_audio: Path
    hypothesis_audio: Path


def _validate_feature(feature: Any, name: str) -> torch.Tensor:
    if (
        not isinstance(feature, torch.Tensor)
        or feature.shape != (4096,)
        or not torch.is_floating_point(feature)
        or not torch.isfinite(feature).all()
    ):
        raise ValueError(f"{name} must be a finite floating tensor with shape [4096].")
    return feature


def _validate_audio(path: Path, source: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing resampled WAV: {path}")
    info = sf.info(path)
    if info.format != "WAV" or info.channels != 1 or info.samplerate != TARGET_SAMPLE_RATE or info.frames < 1:
        raise ValueError(f"Expected a nonempty mono 16 kHz WAV: {path}")
    source_info = sf.info(source)
    expected_frames = (
        source_info.frames
        if source_info.samplerate == TARGET_SAMPLE_RATE
        else (source_info.frames + 2) // 3
    )
    if (
        source_info.format != "WAV"
        or source_info.channels != 1
        or source_info.frames < 1
        or source_info.samplerate not in {TARGET_SAMPLE_RATE, 48_000}
        or info.frames != expected_frames
    ):
        raise ValueError(f"Resampled WAV duration does not match its source: {path}")
    waveform, _ = sf.read(path, dtype="float64", always_2d=False)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError(f"Resampled WAV contains invalid samples: {path}")


def prepare_split(
    raw_data_root: Path,
    source_audio_root: Path,
    output_audio_root: Path,
    old_feature_root: Path,
    new_feature_root: Path,
    split: str,
    *,
    overwrite: bool = False,
) -> tuple[list[PreparedRow], list[Path]]:
    old_root = old_feature_root.resolve()
    new_root = new_feature_root.resolve()
    if old_root == new_root or old_root in new_root.parents or new_root in old_root.parents:
        raise ValueError(f"Old and new feature roots must be separate: {old_root}, {new_root}")
    raw_root = raw_data_root.resolve()
    if raw_root == new_root or raw_root in new_root.parents or new_root in raw_root.parents:
        raise ValueError(f"New feature root must be separate from raw data root: {new_root}")
    source_root = source_audio_root.resolve()
    if source_root == new_root or source_root in new_root.parents or new_root in source_root.parents:
        raise ValueError(f"New feature root must be separate from source audio root: {new_root}")
    audio_root = output_audio_root.resolve()
    if source_root == audio_root or source_root in audio_root.parents or audio_root in source_root.parents:
        raise ValueError(f"Source and resampled audio roots must be separate: {source_root}, {audio_root}")
    if audio_root == new_root or audio_root in new_root.parents or new_root in audio_root.parents:
        raise ValueError(f"New feature root must be separate from resampled audio root: {new_root}")
    output_path = new_root / f"{split}.pt"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Feature output already exists: {output_path}")

    dataset = load_split_dataset(raw_data_root, source_audio_root, split)
    source_split_root = source_audio_root / f"{split}_audio"
    output_split_root = output_audio_root / f"{split}_audio"
    jobs = collect_audio_jobs(dataset, source_split_root, output_split_root)
    for (_, source), path in jobs.items():
        _validate_audio(path, source)

    old_path = old_root / f"{split}.pt"
    if not old_path.is_file():
        raise FileNotFoundError(f"Missing original SONAR features: {old_path}")
    old_samples = torch.load(old_path, map_location="cpu", weights_only=False)
    if not isinstance(old_samples, list) or len(old_samples) != len(dataset):
        raise ValueError(
            f"Original feature row count differs from {split} JSON: "
            f"features={len(old_samples) if isinstance(old_samples, list) else 'not a list'}, "
            f"JSON={len(dataset)}."
        )

    rows: list[PreparedRow] = []
    for index, old in enumerate(old_samples):
        sample = dataset[index]
        if not isinstance(old, dict):
            raise ValueError(f"Original feature row {index} is not a dictionary.")
        sample_index = old.get("sample_index")
        if type(sample_index) is not int or sample_index != index or sample_index != sample["sample_index"]:
            raise ValueError(
                f"{split} row {index}: missing or mismatched sample_index={sample_index!r}."
            )
        if old.get("id") != sample["id"] or old.get("label") != sample["label"]:
            raise ValueError(
                f"{split} row {index}: id/label mismatch between JSON and original features."
            )
        features = old.get("features")
        if not isinstance(features, dict) or set(features) != set(INPUT_MODE_NAMES):
            raise ValueError(f"{split} row {index}: expected exactly four feature modes.")
        text_text = _validate_feature(features[InputMode.TEXT_TEXT.value], f"{split} row {index} TT")
        pair = sample["input_pairs"][InputMode.SPEECH_SPEECH.value]
        premise_source = Path(pair["premise"]).resolve(strict=True)
        hypothesis_source = Path(pair["hypothesis"]).resolve(strict=True)
        premise_audio = jobs[("premise", premise_source)]
        hypothesis_audio = jobs[("hypothesis", hypothesis_source)]
        rows.append(
            PreparedRow(
                id=sample["id"],
                label=sample["label"],
                sample_index=index,
                text_text=text_text,
                premise_audio=premise_audio,
                hypothesis_audio=hypothesis_audio,
            )
        )
    return rows, list(jobs.values())


def load_speech_pipeline(device: str | torch.device) -> Any:
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    except ImportError as exc:
        raise ImportError("SONAR speech pipeline is not installed.") from exc
    return SpeechToEmbeddingModelPipeline(
        encoder="sonar_speech_encoder_vie",
        device=resolve_device(device),
    )


def encode_speech_paths(
    paths: list[Path], pipeline: Any, batch_size: int, *, split: str
) -> dict[Path, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("speech_batch_size must be positive.")
    embeddings: dict[Path, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            result = pipeline.predict([str(path) for path in batch_paths], batch_size=batch_size)
            tensor = torch.as_tensor(result).detach().to(device="cpu", dtype=torch.float32)
            if tensor.shape != (len(batch_paths), 1024) or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{split}: speech encoder must return finite [{len(batch_paths)}, 1024], "
                    f"got {tuple(tensor.shape)}."
                )
            embeddings.update({path: tensor[i].clone() for i, path in enumerate(batch_paths)})
            if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(paths):
                print(f"{split}: encoded {min(start + batch_size, len(paths))}/{len(paths)} WAVs", flush=True)
    return embeddings


def build_rebuilt_features(
    rows: list[PreparedRow], speech_embeddings: dict[Path, torch.Tensor]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        text_premise, text_hypothesis = split_pair_feature(row.text_text)
        speech_premise = speech_embeddings[row.premise_audio]
        speech_hypothesis = speech_embeddings[row.hypothesis_audio]
        features = {
            InputMode.TEXT_TEXT.value: row.text_text,
            InputMode.TEXT_SPEECH.value: build_pair_feature(text_premise, speech_hypothesis),
            InputMode.SPEECH_TEXT.value: build_pair_feature(speech_premise, text_hypothesis),
            InputMode.SPEECH_SPEECH.value: build_pair_feature(speech_premise, speech_hypothesis),
        }
        if tuple(features) != INPUT_MODE_NAMES:
            raise ValueError(f"Unexpected feature mode order for sample_index={row.sample_index}.")
        for mode, feature in features.items():
            _validate_feature(feature, f"sample_index={row.sample_index} {mode}")
        results.append(
            NLIFeatureSample(
                id=row.id,
                label=row.label,
                sample_index=row.sample_index,
                features=features,
            ).to_dict()
        )
    return results


def save_features_atomic(
    features: list[dict[str, Any]], output_path: Path, *, overwrite: bool = False
) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Feature output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output_path.stem}.", suffix=".pt", dir=output_path.parent)
    os.close(fd)
    temp_path = Path(name)
    try:
        torch.save(features, temp_path)
        reloaded = torch.load(temp_path, map_location="cpu", weights_only=False)
        if not isinstance(reloaded, list) or len(reloaded) != len(features):
            raise ValueError(f"Serialized feature row count mismatch: {temp_path}")
        for index, (before, after) in enumerate(zip(features, reloaded)):
            if (
                not isinstance(after, dict)
                or after.get("id") != before["id"]
                or after.get("label") != before["label"]
                or after.get("sample_index") != index
            ):
                raise ValueError(f"Serialized feature metadata mismatch at row {index}.")
            modes = after.get("features")
            if not isinstance(modes, dict) or tuple(modes) != INPUT_MODE_NAMES:
                raise ValueError(f"Serialized feature modes mismatch at row {index}.")
            for mode in INPUT_MODE_NAMES:
                _validate_feature(modes[mode], f"serialized row {index} {mode}")
                if not torch.equal(modes[mode], before["features"][mode]):
                    raise ValueError(f"Serialized {mode} changed at row {index}.")
        del reloaded
        publish_temp_file(temp_path, output_path, overwrite=overwrite)
    finally:
        temp_path.unlink(missing_ok=True)


def _rebuild_prepared_split(
    rows: list[PreparedRow],
    paths: list[Path],
    split: str,
    new_feature_root: Path,
    speech_pipeline: Any,
    speech_batch_size: int,
    overwrite: bool,
) -> int:
    embeddings = encode_speech_paths(paths, speech_pipeline, speech_batch_size, split=split)
    features = build_rebuilt_features(rows, embeddings)
    save_features_atomic(features, new_feature_root / f"{split}.pt", overwrite=overwrite)
    print(f"{split}: saved {len(features)} four-mode samples to {new_feature_root / f'{split}.pt'}", flush=True)
    return len(features)


def rebuild_split(
    raw_data_root: Path,
    source_audio_root: Path,
    output_audio_root: Path,
    old_feature_root: Path,
    new_feature_root: Path,
    split: str,
    *,
    speech_pipeline: Any,
    speech_batch_size: int = 8,
    overwrite: bool = False,
) -> int:
    rows, paths = prepare_split(
        raw_data_root,
        source_audio_root,
        output_audio_root,
        old_feature_root,
        new_feature_root,
        split,
        overwrite=overwrite,
    )
    return _rebuild_prepared_split(
        rows, paths, split, new_feature_root, speech_pipeline, speech_batch_size, overwrite
    )


def run_rebuild(
    raw_data_root: Path,
    source_audio_root: Path,
    output_audio_root: Path,
    old_feature_root: Path,
    new_feature_root: Path,
    splits: list[str],
    *,
    device: str = "auto",
    speech_batch_size: int = 8,
    overwrite: bool = False,
) -> None:
    if speech_batch_size < 1:
        raise ValueError("speech_batch_size must be positive.")
    if len(splits) != len(set(splits)):
        raise ValueError(f"Duplicate split requested: {splits}")
    pipeline: Any | None = None
    for split in splits:
        # Validate alignment and audio before loading the speech model for this split.
        rows, paths = prepare_split(
            raw_data_root,
            source_audio_root,
            output_audio_root,
            old_feature_root,
            new_feature_root,
            split,
            overwrite=overwrite,
        )
        if pipeline is None:
            pipeline = load_speech_pipeline(device)
        _rebuild_prepared_split(
            rows, paths, split, new_feature_root, pipeline, speech_batch_size, overwrite
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild SONAR features using only 16 kHz speech WAVs.")
    parser.add_argument("--raw-data-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--source-audio-root", type=Path, default=None)
    parser.add_argument(
        "--output-audio-root", type=Path, default=PROJECT_ROOT / "Dataset_resampled_16k"
    )
    parser.add_argument("--old-feature-root", type=Path, default=PROJECT_ROOT / "Features" / "sonar")
    parser.add_argument("--new-feature-root", type=Path, default=PROJECT_ROOT / "Features" / "sonar_16k")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--speech-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_rebuild(
        args.raw_data_root,
        args.source_audio_root or args.raw_data_root,
        args.output_audio_root,
        args.old_feature_root,
        args.new_feature_root,
        args.splits,
        device=args.device,
        speech_batch_size=args.speech_batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
