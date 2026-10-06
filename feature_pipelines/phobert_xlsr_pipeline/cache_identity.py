from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict
from pathlib import Path

from ...data.dataset import NLIMultimodalDataset
from .config import PhoBERTXLSRConfig
from .schemas import CacheModality


def sentence_key(text: str) -> str:
    """Identify equal sentences independently of NLI id and label."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sentence identity requires non-empty text.")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compute_cache_fingerprint(
    config: PhoBERTXLSRConfig,
    split: str,
    modality: CacheModality,
) -> str:
    """Bind a cache to source contents, resolved audio and encoder preprocessing."""

    source = config.data.split_path(split)
    dataset = NLIMultimodalDataset(
        source,
        validate_audio_exists=False,
        allow_ambiguous_audio=config.data.allow_ambiguous_audio,
    )
    audio_paths = {Path(path).resolve() for path in dataset.audio_index.values()}
    for record in dataset.records:
        for side in ("premise", "hypothesis"):
            value = record.get(f"{side}_audio")
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = dataset.audio_root / path
                audio_paths.add(path.resolve())
    encoder = config.text_encoder if modality == CacheModality.TEXT else config.speech_encoder
    settings = {
        "cache_schema_version": 2,
        "source_sha256": _file_digest(source),
        "source_path": str(source.resolve()),
        "encoder": asdict(encoder),
        "mixed_precision": config.mixed_precision,
        "seed": config.seed,
        "data_policy": {
            "missing_audio_policy": config.data.missing_audio_policy,
            "validate_audio_exists": config.data.validate_audio_exists,
            "allow_ambiguous_audio": config.data.allow_ambiguous_audio,
        },
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    # Text loading also depends on audio availability, but not on waveform contents.
    for path in sorted(audio_paths):
        signature = "missing"
        if path.is_file():
            signature = _file_digest(path) if modality == CacheModality.SPEECH else "present"
        digest.update(json.dumps([str(path), signature]).encode("utf-8"))
    return digest.hexdigest()
