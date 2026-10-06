from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import AudioOverlengthPolicy, PAIR_FEATURE_COMPONENTS


KLTN_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DataSettings:
    raw_data_dir: Path
    split_files: dict[str, Path]
    num_workers: int
    pin_memory: bool
    validate_audio_exists: bool
    allow_ambiguous_audio: bool
    missing_audio_policy: str

    def split_path(self, split: str) -> Path:
        if split not in self.split_files:
            available = ", ".join(sorted(self.split_files))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}.")
        return self.raw_data_dir / self.split_files[split]


@dataclass(frozen=True)
class TextEncoderSettings:
    model_name: str
    max_length: int
    pooling: str
    batch_size: int
    freeze: bool
    use_word_segmentation: bool
    cache_segmented_text: bool


@dataclass(frozen=True)
class SpeechEncoderSettings:
    model_name: str
    target_sample_rate: int
    max_audio_seconds: float
    overlength_policy: AudioOverlengthPolicy
    pooling: str
    batch_size: int
    freeze: bool


@dataclass(frozen=True)
class ProjectorSettings:
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str
    dropout: float


@dataclass(frozen=True)
class AlignmentSettings:
    enabled: bool
    temperature: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int


@dataclass(frozen=True)
class PairFeatureSettings:
    components: tuple[str, ...]
    output_dim: int


@dataclass(frozen=True)
class CacheSettings:
    directory: Path
    shard_size: int
    dtype: str
    overwrite: bool


@dataclass(frozen=True)
class CheckpointSettings:
    directory: Path
    save_best_only: bool


@dataclass(frozen=True)
class OutputSettings:
    directory: Path
    dtype: str
    save_metadata: bool
    overwrite: bool


@dataclass(frozen=True)
class PhoBERTXLSRConfig:
    pipeline_name: str
    seed: int
    device: str
    mixed_precision: bool
    data: DataSettings
    text_encoder: TextEncoderSettings
    speech_encoder: SpeechEncoderSettings
    projector: ProjectorSettings
    alignment: AlignmentSettings
    pair_feature: PairFeatureSettings
    cache: CacheSettings
    checkpoint: CheckpointSettings
    output: OutputSettings

    def to_dict(self) -> dict[str, Any]:
        return _serialize_paths(asdict(self))


def load_pipeline_config(path: str | Path) -> PhoBERTXLSRConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    data = _section(payload, "data")
    text = _section(payload, "text_encoder")
    speech = _section(payload, "speech_encoder")
    projector = _section(payload, "projector")
    alignment = _section(payload, "alignment")
    pair_feature = _section(payload, "pair_feature")
    cache = _section(payload, "cache")
    checkpoint = _section(payload, "checkpoint")
    output = _section(payload, "output")

    config = PhoBERTXLSRConfig(
        pipeline_name=str(payload.get("pipeline_name", "phobert_xlsr")),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "auto")),
        mixed_precision=bool(payload.get("mixed_precision", True)),
        data=DataSettings(
            raw_data_dir=_resolve_path(data["raw_data_dir"]),
            split_files={name: Path(value) for name, value in data["split_files"].items()},
            num_workers=int(data.get("num_workers", 0)),
            pin_memory=bool(data.get("pin_memory", False)),
            validate_audio_exists=bool(data.get("validate_audio_exists", True)),
            allow_ambiguous_audio=bool(data.get("allow_ambiguous_audio", True)),
            missing_audio_policy=str(data.get("missing_audio_policy", "raise")),
        ),
        text_encoder=TextEncoderSettings(
            model_name=str(text["model_name"]),
            max_length=int(text["max_length"]),
            pooling=str(text["pooling"]),
            batch_size=int(text["batch_size"]),
            freeze=bool(text.get("freeze", True)),
            use_word_segmentation=bool(text.get("use_word_segmentation", True)),
            cache_segmented_text=bool(text.get("cache_segmented_text", True)),
        ),
        speech_encoder=SpeechEncoderSettings(
            model_name=str(speech["model_name"]),
            target_sample_rate=int(speech["target_sample_rate"]),
            max_audio_seconds=float(speech["max_audio_seconds"]),
            overlength_policy=AudioOverlengthPolicy(speech["overlength_policy"]),
            pooling=str(speech["pooling"]),
            batch_size=int(speech["batch_size"]),
            freeze=bool(speech.get("freeze", True)),
        ),
        projector=ProjectorSettings(
            input_dim=int(projector["input_dim"]),
            hidden_dim=int(projector["hidden_dim"]),
            output_dim=int(projector["output_dim"]),
            activation=str(projector.get("activation", "gelu")),
            dropout=float(projector["dropout"]),
        ),
        alignment=AlignmentSettings(
            enabled=bool(alignment.get("enabled", True)),
            temperature=float(alignment["temperature"]),
            batch_size=int(alignment["batch_size"]),
            learning_rate=float(alignment["learning_rate"]),
            weight_decay=float(alignment["weight_decay"]),
            epochs=int(alignment["epochs"]),
            patience=int(alignment["patience"]),
        ),
        pair_feature=PairFeatureSettings(
            components=tuple(pair_feature["components"]),
            output_dim=int(pair_feature["output_dim"]),
        ),
        cache=CacheSettings(
            directory=_resolve_path(cache["directory"]),
            shard_size=int(cache["shard_size"]),
            dtype=str(cache["dtype"]),
            overwrite=bool(cache.get("overwrite", False)),
        ),
        checkpoint=CheckpointSettings(
            directory=_resolve_path(checkpoint["directory"]),
            save_best_only=bool(checkpoint.get("save_best_only", True)),
        ),
        output=OutputSettings(
            directory=_resolve_path(output["directory"]),
            dtype=str(output["dtype"]),
            save_metadata=bool(output.get("save_metadata", True)),
            overwrite=bool(output.get("overwrite", False)),
        ),
    )
    _validate_config(config)
    return config


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == KLTN_ROOT.name.lower():
        return KLTN_ROOT.parent / path
    return KLTN_ROOT / path


def _validate_config(config: PhoBERTXLSRConfig) -> None:
    if (
        config.text_encoder.pooling != "masked_mean"
        or config.speech_encoder.pooling != "masked_mean"
    ):
        raise ValueError("The initial pipeline supports only masked_mean pooling.")
    if not config.text_encoder.freeze or not config.speech_encoder.freeze:
        raise ValueError("The initial pipeline requires both base encoders to remain frozen.")
    if config.projector.activation != "gelu":
        raise ValueError("The initial pipeline supports only GELU projectors.")
    if not config.alignment.enabled:
        raise ValueError("alignment.enabled must be true for the initial aligned feature pipeline.")
    if config.projector.input_dim != 1024 or config.projector.output_dim != 1024:
        raise ValueError("Projector input_dim and output_dim must both be 1024.")
    if config.projector.hidden_dim <= 0:
        raise ValueError("projector.hidden_dim must be positive.")
    if not 0.0 <= config.projector.dropout < 1.0:
        raise ValueError("projector.dropout must be in [0, 1).")
    if config.pair_feature.components != PAIR_FEATURE_COMPONENTS:
        raise ValueError(f"pair_feature.components must be {PAIR_FEATURE_COMPONENTS}.")
    if config.pair_feature.output_dim != 4 * config.projector.output_dim:
        raise ValueError("pair_feature.output_dim must equal four times projector.output_dim.")
    positive_values = {
        "text max_length": config.text_encoder.max_length,
        "text batch_size": config.text_encoder.batch_size,
        "speech target_sample_rate": config.speech_encoder.target_sample_rate,
        "speech max_audio_seconds": config.speech_encoder.max_audio_seconds,
        "speech batch_size": config.speech_encoder.batch_size,
        "cache shard_size": config.cache.shard_size,
        "alignment batch_size": config.alignment.batch_size,
        "alignment temperature": config.alignment.temperature,
        "alignment learning_rate": config.alignment.learning_rate,
        "alignment epochs": config.alignment.epochs,
        "alignment patience": config.alignment.patience,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if config.alignment.batch_size < 2:
        raise ValueError("alignment.batch_size must be at least 2.")
    if config.alignment.weight_decay < 0:
        raise ValueError("alignment.weight_decay must be non-negative.")
    if config.data.missing_audio_policy not in {"raise", "skip"}:
        raise ValueError("data.missing_audio_policy must be 'raise' or 'skip'.")
    supported_dtypes = {"float32", "float16", "bfloat16"}
    if config.cache.dtype not in supported_dtypes:
        raise ValueError(f"cache.dtype must be one of {sorted(supported_dtypes)}.")
    if config.output.dtype not in supported_dtypes:
        raise ValueError(f"output.dtype must be one of {sorted(supported_dtypes)}.")
    if not config.output.save_metadata:
        raise ValueError("output.save_metadata must be true for auditable feature files.")
    required_splits = {"train", "dev", "test"}
    missing_splits = required_splits - config.data.split_files.keys()
    if missing_splits:
        raise ValueError(f"data.split_files is missing required splits: {sorted(missing_splits)}.")


def _serialize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_paths(item) for item in value]
    if isinstance(value, AudioOverlengthPolicy):
        return value.value
    return value
