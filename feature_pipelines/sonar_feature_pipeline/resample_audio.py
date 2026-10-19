from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ...data.dataset import NLIMultimodalDataset
from ...data.schemas import InputMode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "dev", "test")
TARGET_SAMPLE_RATE = 16_000
SOURCE_SAMPLE_RATE = 48_000
AudioJobs = dict[tuple[str, Path], Path]


def split_data_path(raw_data_root: Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}.")
    candidates = [raw_data_root / f"{split}{suffix}" for suffix in (".json", ".jsonl")]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"Expected exactly one JSON/JSONL file for {split} in {raw_data_root}; "
            f"found {present}."
        )
    return present[0]


def load_split_dataset(
    raw_data_root: Path, source_audio_root: Path, split: str
) -> NLIMultimodalDataset:
    dataset = NLIMultimodalDataset(
        split_data_path(raw_data_root, split),
        input_mode=InputMode.TEXT_TEXT,
        audio_root=source_audio_root / f"{split}_audio",
        validate_audio_exists=True,
        allow_ambiguous_audio=False,
    )
    if len(dataset) == 0:
        raise ValueError(f"Split {split!r} has no NLI samples.")
    return dataset


def output_path_for_audio(
    source_path: str | Path,
    side: str,
    source_split_root: Path,
    output_split_root: Path,
) -> Path:
    if side not in {"premise", "hypothesis"}:
        raise ValueError(f"Unsupported audio side: {side!r}.")
    source = Path(source_path).resolve(strict=True)
    root = source_split_root.resolve(strict=True)
    if root not in source.parents:
        raise ValueError(f"Audio source is outside declared audio root {root}: {source}")
    if not source.is_file() or source.suffix.lower() != ".wav":
        raise ValueError(f"Audio source must be a WAV file: {source}")
    destination = (output_split_root / f"{side}_audio" / source.name).resolve()
    if destination == source:
        raise ValueError(f"Source and output WAV paths are identical: {source}")
    return destination


def collect_audio_jobs(
    dataset: NLIMultimodalDataset,
    source_split_root: Path,
    output_split_root: Path,
) -> AudioJobs:
    source_root = source_split_root.resolve(strict=True)
    output_root = output_split_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError(
            f"Source and output audio roots must be separate: {source_root}, {output_root}"
        )

    jobs: AudioJobs = {}
    destination_sources: dict[Path, Path] = {}
    for index in range(len(dataset)):
        pair = dataset[index]["input_pairs"][InputMode.SPEECH_SPEECH.value]
        for side in ("premise", "hypothesis"):
            source = Path(pair[side]).resolve(strict=True)
            destination = output_path_for_audio(source, side, source_root, output_root)
            previous = destination_sources.get(destination)
            if previous is not None and previous != source:
                raise ValueError(
                    f"Audio output collision at {destination}: {previous} and {source}"
                )
            destination_sources[destination] = source
            jobs[(side, source)] = destination
    return jobs


def _read_source(path: Path) -> tuple[np.ndarray, Any]:
    info = sf.info(path)
    if info.format != "WAV" or info.channels != 1:
        raise ValueError(f"Expected a mono WAV, got {info.format}/{info.channels} channels: {path}")
    if info.samplerate not in {SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE}:
        raise ValueError(
            f"Expected 48000 or 16000 Hz, got {info.samplerate} Hz: {path}"
        )
    waveform, rate = sf.read(path, dtype="float64", always_2d=False)
    if rate != info.samplerate or waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"Empty or invalid mono WAV: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"WAV contains NaN or Inf: {path}")
    return waveform, info


def _output_subtype(source_subtype: str, waveform: np.ndarray) -> str:
    pcm_bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
    if source_subtype in pcm_bits:
        upper = 1.0 - 2.0 ** (1 - pcm_bits[source_subtype])
        if waveform.min() >= -1.0 and waveform.max() <= upper:
            return source_subtype
    if source_subtype in {"FLOAT", "DOUBLE"}:
        return source_subtype
    # FIR ringing can exceed the representable PCM range; FLOAT avoids clipping.
    return "FLOAT"


def _check_output(
    path: Path,
    source_path: Path,
    source_info: Any,
    expected: np.ndarray,
    subtype: str,
) -> None:
    info = sf.info(path)
    if (
        info.format != "WAV"
        or info.channels != 1
        or info.samplerate != TARGET_SAMPLE_RATE
        or info.frames != len(expected)
        or info.subtype != subtype
    ):
        raise ValueError(f"Output WAV metadata does not match source conversion: {path}")
    if abs(info.duration - source_info.duration) > 1 / TARGET_SAMPLE_RATE + 1e-9:
        raise ValueError(f"Output WAV duration differs from source: {path}")
    actual, _ = sf.read(path, dtype="float64", always_2d=False)
    if not np.isfinite(actual).all():
        raise ValueError(f"Output WAV contains NaN or Inf: {path}")
    if source_info.samplerate == TARGET_SAMPLE_RATE:
        if path.read_bytes() != source_path.read_bytes():
            raise ValueError(f"16 kHz output must be a byte-for-byte copy: {path}")
    else:
        pcm_step = {"PCM_16": 2.0**-15, "PCM_24": 2.0**-23, "PCM_32": 2.0**-31}
        tolerance = pcm_step.get(subtype, 2e-6 if subtype == "FLOAT" else 1e-12)
        if not np.allclose(actual, expected, rtol=0, atol=tolerance):
            raise ValueError(f"Output WAV samples do not match resampling result: {path}")


def publish_temp_file(temp_path: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish a completed file without clobbering an existing output by default."""
    if overwrite:
        os.replace(temp_path, destination)
    else:
        os.link(temp_path, destination)
        temp_path.unlink()


def resample_wav(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> str:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError(f"Source and output WAV paths are identical: {source}")
    waveform, info = _read_source(source)
    if info.samplerate == TARGET_SAMPLE_RATE:
        expected = waveform
        subtype = info.subtype
    else:
        expected = resample_poly(waveform, up=1, down=3)
        if not np.isfinite(expected).all():
            raise ValueError(f"Resampling produced NaN or Inf: {source}")
        subtype = _output_subtype(info.subtype, expected)

    if destination.exists():
        if resume:
            _check_output(destination, source, info, expected, subtype)
            return "skipped"
        if not overwrite:
            raise FileExistsError(f"Output WAV already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".wav", dir=destination.parent)
    os.close(fd)
    temp_path = Path(name)
    try:
        if info.samplerate == TARGET_SAMPLE_RATE:
            shutil.copyfile(source, temp_path)
        else:
            sf.write(temp_path, expected, TARGET_SAMPLE_RATE, format="WAV", subtype=subtype)
        _check_output(temp_path, source, info, expected, subtype)
        publish_temp_file(temp_path, destination, overwrite=overwrite)
    finally:
        temp_path.unlink(missing_ok=True)
    return "written"


def run_resampling(
    raw_data_root: Path,
    source_audio_root: Path,
    output_audio_root: Path,
    splits: list[str],
    *,
    target_rate: int = TARGET_SAMPLE_RATE,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    if target_rate != TARGET_SAMPLE_RATE:
        raise ValueError("Only 16000 Hz output is supported.")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    if len(splits) != len(set(splits)):
        raise ValueError(f"Duplicate split requested: {splits}")
    raw_root = raw_data_root.resolve()
    source_root = source_audio_root.resolve()
    output_root = output_audio_root.resolve()
    if raw_root == output_root or raw_root in output_root.parents or output_root in raw_root.parents:
        raise ValueError(f"Output audio root must be separate from raw data root: {output_root}")
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError(f"Output audio root must be separate from source audio root: {output_root}")
    for split in splits:
        dataset = load_split_dataset(raw_data_root, source_audio_root, split)
        jobs = collect_audio_jobs(
            dataset,
            source_audio_root / f"{split}_audio",
            output_audio_root / f"{split}_audio",
        )
        written = skipped = 0
        for (_, source), destination in jobs.items():
            result = resample_wav(source, destination, overwrite=overwrite, resume=resume)
            written += result == "written"
            skipped += result == "skipped"
        print(f"{split}: samples={len(dataset)} WAVs={len(jobs)} written={written} skipped={skipped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample referenced NLI WAVs to 16 kHz.")
    parser.add_argument("--raw-data-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--source-audio-root", type=Path, default=None)
    parser.add_argument(
        "--output-audio-root", type=Path, default=PROJECT_ROOT / "Dataset_resampled_16k"
    )
    parser.add_argument("--target-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_resampling(
        args.raw_data_root,
        args.source_audio_root or args.raw_data_root,
        args.output_audio_root,
        args.splits,
        target_rate=args.target_rate,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
