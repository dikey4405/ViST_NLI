from __future__ import annotations

import argparse
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ...data.collator import NLICollator
from ...data.schemas import INPUT_MODE_NAMES, InputMode
from ...pair_features import build_pair_feature
from .config import FullSonar16kConfig, SAMPLE_RATE, SPLITS
from .dataset import FullSonar16kDataset
from .sonar_encoder import FullSonarEncoder


def _encode_unique(values: list[str], encode: Any, kind: str) -> dict[str, torch.Tensor]:
    unique_values = list(dict.fromkeys(values))
    embeddings = torch.as_tensor(encode(unique_values)).detach().to(
        device="cpu", dtype=torch.float32
    )
    if embeddings.shape != (len(unique_values), 1024) or not torch.isfinite(embeddings).all():
        raise ValueError(
            f"{kind} encoder must return finite [{len(unique_values)}, 1024] embeddings, "
            f"got {tuple(embeddings.shape)}."
        )
    return dict(zip(unique_values, embeddings))


def extract_batch_features(batch: dict[str, Any], encoder: Any) -> list[dict[str, Any]]:
    """Build all four modes from fresh text and speech embeddings for one batch."""
    pairs = batch["input_pairs"]
    text_pair = pairs[InputMode.TEXT_TEXT.value]
    speech_pair = pairs[InputMode.SPEECH_SPEECH.value]
    batch_size = len(batch["ids"])
    if any(
        len(values) != batch_size
        for values in (
            text_pair["premise"],
            text_pair["hypothesis"],
            speech_pair["premise"],
            speech_pair["hypothesis"],
        )
    ):
        raise ValueError("Input pair lengths do not match batch metadata.")

    text_embeddings = _encode_unique(
        text_pair["premise"] + text_pair["hypothesis"], encoder.encode_text, "Text"
    )
    speech_embeddings = _encode_unique(
        speech_pair["premise"] + speech_pair["hypothesis"], encoder.encode_speech, "Speech"
    )
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        text_premise = text_embeddings[text_pair["premise"][index]]
        text_hypothesis = text_embeddings[text_pair["hypothesis"][index]]
        speech_premise = speech_embeddings[speech_pair["premise"][index]]
        speech_hypothesis = speech_embeddings[speech_pair["hypothesis"][index]]
        features = {
            InputMode.TEXT_TEXT.value: build_pair_feature(text_premise, text_hypothesis),
            InputMode.TEXT_SPEECH.value: build_pair_feature(text_premise, speech_hypothesis),
            InputMode.SPEECH_TEXT.value: build_pair_feature(speech_premise, text_hypothesis),
            InputMode.SPEECH_SPEECH.value: build_pair_feature(speech_premise, speech_hypothesis),
        }
        if tuple(features) != INPUT_MODE_NAMES:
            raise ValueError("Unexpected feature mode order.")
        for mode, feature in features.items():
            if (
                feature.shape != (4096,)
                or feature.dtype != torch.float32
                or not torch.isfinite(feature).all()
            ):
                raise ValueError(
                    f"sample_index={batch['sample_indices'][index]}, mode={mode}: "
                    "feature must be finite float32 [4096]."
                )
        rows.append(
            {
                "id": batch["ids"][index],
                "label": batch["labels"][index],
                "sample_index": batch["sample_indices"][index],
                "features": features,
            }
        )
    return rows


def _validate_saved_rows(saved: Any, original: list[dict[str, Any]]) -> None:
    if not isinstance(saved, list) or len(saved) != len(original):
        raise ValueError("Serialized feature row count does not match the input split.")
    for index, (before, after) in enumerate(zip(original, saved)):
        if not isinstance(after, dict) or (
            after.get("id"), after.get("label"), after.get("sample_index")
        ) != (before["id"], before["label"], index):
            raise ValueError(f"Serialized metadata mismatch at sample_index={index}.")
        features = after.get("features")
        if not isinstance(features, dict) or tuple(features) != INPUT_MODE_NAMES:
            raise ValueError(f"Serialized mode order mismatch at sample_index={index}.")
        for mode in INPUT_MODE_NAMES:
            feature = features[mode]
            if (
                not isinstance(feature, torch.Tensor)
                or feature.shape != (4096,)
                or feature.dtype != torch.float32
                or not torch.isfinite(feature).all()
                or not torch.equal(feature, before["features"][mode])
            ):
                raise ValueError(
                    f"Serialized feature mismatch at sample_index={index}, mode={mode}."
                )


def save_features_atomic(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Validate a temporary .pt and publish without replacing an existing output."""
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Feature output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".pt.tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(name)
    try:
        torch.save(rows, temp_path)
        saved = torch.load(temp_path, map_location="cpu", weights_only=False)
        _validate_saved_rows(saved, rows)
        del saved
        os.link(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _sonar_version() -> str:
    try:
        return version("sonar-space")
    except PackageNotFoundError:
        return "unavailable"


def run_extraction(config: FullSonar16kConfig, *, encoder: Any | None = None) -> dict[str, Path]:
    """Preflight selected splits, then encode and save each split independently."""
    datasets: dict[str, FullSonar16kDataset] = {}
    for split in config.splits:
        output_path = config.output_dir / f"{split}.pt"
        if output_path.exists():
            raise FileExistsError(f"Feature output already exists: {output_path}")
        dataset = FullSonar16kDataset(config.split_path(split), config.audio_root_16k, split)
        datasets[split] = dataset
        print(
            f"PREFLIGHT | split={split} | samples={len(dataset)} | "
            f"unique_wavs={dataset.resolver.unique_wav_count} | sample_rate={SAMPLE_RATE}",
            flush=True,
        )

    if encoder is None:
        encoder = FullSonarEncoder(config)
    print(
        f"ENCODERS | text={config.text_encoder} | tokenizer={config.text_tokenizer} | "
        f"lang={config.text_source_lang} | speech={config.speech_encoder} | "
        f"sonar_space={_sonar_version()} | device={getattr(encoder, 'device', config.device)}",
        flush=True,
    )

    output_paths: dict[str, Path] = {}
    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=NLICollator(),
        )
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                rows.extend(extract_batch_features(batch, encoder))
                if batch_index % 25 == 0 or batch_index == len(loader):
                    print(
                        f"PROGRESS | split={split} | batches={batch_index}/{len(loader)} | "
                        f"samples={len(rows)}/{len(dataset)}",
                        flush=True,
                    )
        if len(rows) != len(dataset) or any(
            row["sample_index"] != index for index, row in enumerate(rows)
        ):
            raise ValueError(f"Split {split}: extracted rows differ from the raw JSON order/count.")
        output_path = config.output_dir / f"{split}.pt"
        save_features_atomic(rows, output_path)
        output_paths[split] = output_path
        print(
            f"SAVED | split={split} | samples={len(rows)} | "
            f"unique_wavs={dataset.resolver.unique_wav_count} | path={output_path}",
            flush=True,
        )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full SONAR NLI features from raw text and 16 kHz WAVs."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--audio-root-16k", type=Path, default=Path("Dataset_resampled_16k"))
    parser.add_argument("--output-dir", type=Path, default=Path("Features/sonar_full_16k"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    config = FullSonar16kConfig(
        raw_data_dir=args.raw_data_dir,
        audio_root_16k=args.audio_root_16k,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        encoder_batch_size=args.encoder_batch_size,
        device=args.device,
    )
    run_extraction(config)


if __name__ == "__main__":
    main()
