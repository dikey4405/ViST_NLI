from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .build_dataloader import build_dataloader
from .config import FeatureExtractionConfig
from .feature_extractor import build_pair_feature
from .input_builder import build_batch_input_pairs
from .schemas import ModalityType, NLIFeatureSample
from .sonar_encoder import SonarEncoder
from .utils import ensure_output_dir, save_pt, validate_feature_shape


def extract_and_save_features(
    config: FeatureExtractionConfig,
    splits: list[str] | None = None,
) -> None:
    """Extract SONAR pair features for all requested splits and save them as .pt files."""

    output_dir = ensure_output_dir(config.get_output_dir())
    encoder = SonarEncoder(
        text_encoder_name_or_path=config.text_encoder_name_or_path,
        text_tokenizer_name_or_path=config.text_tokenizer_name_or_path,
        speech_encoder_name_or_path=config.speech_encoder_name_or_path,
        text_source_lang=config.text_source_lang,
        batch_size=config.encoder_batch_size,
        device=config.device,
    )

    selected_splits = splits or list(config.split_files.keys())
    for split in selected_splits:
        dataloader = build_dataloader(config, split)
        features = extract_split_features(dataloader, encoder)
        save_pt(features, output_dir / f"{split}.pt")


def extract_split_features(dataloader: Any, encoder: SonarEncoder) -> list[dict[str, Any]]:
    """Extract feature dictionaries for one DataLoader split."""

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_features = extract_batch_features(batch, encoder)
            batch_size = len(batch["ids"])
            for index in range(batch_size):
                sample_features = {
                    mode: mode_features[index].detach().cpu()
                    for mode, mode_features in batch_features.items()
                }
                results.append(
                    NLIFeatureSample(
                        id=batch["ids"][index],
                        label=batch["labels"][index],
                        sample_index=batch["sample_indices"][index],
                        features=sample_features,
                    ).to_dict()
                )
    return results


def extract_batch_features(batch: dict[str, Any], encoder: SonarEncoder) -> dict[str, torch.Tensor]:
    """Build [batch_size, 4096] features for each input mode in a batch."""

    input_pairs = build_batch_input_pairs(batch)
    features: dict[str, torch.Tensor] = {}
    embedding_cache: dict[tuple[str, str, tuple[str, ...]], torch.Tensor] = {}

    for mode, pair_batch in input_pairs.items():
        premise_embeddings = _get_or_encode_values(
            encoder,
            cache=embedding_cache,
            side="premise",
            values=pair_batch["premise"],
            modality=pair_batch["premise_modality"][0],
        )
        hypothesis_embeddings = _get_or_encode_values(
            encoder,
            cache=embedding_cache,
            side="hypothesis",
            values=pair_batch["hypothesis"],
            modality=pair_batch["hypothesis_modality"][0],
        )
        mode_features = build_pair_feature(premise_embeddings, hypothesis_embeddings)
        validate_feature_shape(mode_features, name=f"{mode} feature")
        features[mode] = mode_features

    return features


def _get_or_encode_values(
    encoder: SonarEncoder,
    *,
    cache: dict[tuple[str, str, tuple[str, ...]], torch.Tensor],
    side: str,
    values: list[str],
    modality: str,
) -> torch.Tensor:
    cache_key = (side, modality, tuple(values))
    if cache_key not in cache:
        cache[cache_key] = _encode_values(encoder, values=values, modality=modality)
    return cache[cache_key]


def _encode_values(encoder: SonarEncoder, *, values: list[str], modality: str) -> torch.Tensor:
    if modality == ModalityType.TEXT.value:
        return encoder.encode_text(values)
    if modality == ModalityType.SPEECH.value:
        return encoder.encode_speech(values)
    raise ValueError(f"Unsupported modality: {modality}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SONAR NLI pair features.")
    parser.add_argument(
        "--raw-data-dir",
        "--data-dir",
        dest="raw_data_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to process, e.g. train dev test.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text-encoder", type=str, default=None)
    parser.add_argument("--text-tokenizer", type=str, default=None)
    parser.add_argument("--speech-encoder", type=str, default=None)
    parser.add_argument("--text-source-lang", type=str, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--no-validate-audio", action="store_true")
    parser.add_argument("--strict-audio-ambiguity", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> FeatureExtractionConfig:
    base = FeatureExtractionConfig()
    return FeatureExtractionConfig(
        raw_data_dir=args.raw_data_dir or base.raw_data_dir,
        output_dir=args.output_dir or base.output_dir,
        split_files=base.split_files,
        batch_size=args.batch_size or base.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else base.num_workers,
        device=args.device or base.device,
        text_encoder_name_or_path=args.text_encoder or base.text_encoder_name_or_path,
        text_tokenizer_name_or_path=args.text_tokenizer or base.text_tokenizer_name_or_path,
        speech_encoder_name_or_path=args.speech_encoder or base.speech_encoder_name_or_path,
        text_source_lang=args.text_source_lang or base.text_source_lang,
        encoder_batch_size=args.encoder_batch_size or base.encoder_batch_size,
        label_mapping=base.label_mapping,
        validate_audio_exists=not args.no_validate_audio,
        allow_ambiguous_audio=not args.strict_audio_ambiguity,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    extract_and_save_features(config, splits=args.splits)


if __name__ == "__main__":
    main()
