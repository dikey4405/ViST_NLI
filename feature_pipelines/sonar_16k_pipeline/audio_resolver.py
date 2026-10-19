from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

import soundfile as sf

from .config import SAMPLE_RATE, SPLITS


_AUDIO_PREFIXES = {"premise": "prem", "hypothesis": "hypo"}


class AudioResolver:
    """Map raw audio references to the corresponding split-local 16 kHz WAVs."""

    def __init__(self, audio_root_16k: Path, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split!r}.")
        self.split = split
        self.audio_root = Path(audio_root_16k).resolve()
        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"Missing 16 kHz audio root: {self.audio_root}")
        self.split_root = self.audio_root / f"{split}_audio"
        self._validated: set[Path] = set()
        self._assignments: dict[tuple[str, Path], tuple[str | None, str]] = {}

    @property
    def unique_wav_count(self) -> int:
        return len(self._validated)

    def resolve(
        self,
        record: dict[str, Any],
        *,
        sample_index: int,
        sample_id: str,
        label: str,
        side: str,
    ) -> str:
        if side not in _AUDIO_PREFIXES:
            raise ValueError(f"Unsupported audio side: {side!r}.")
        context = f"split={self.split}, sample_index={sample_index}, id={sample_id!r}, side={side}"
        raw_reference = record.get(f"{side}_audio")
        if raw_reference is not None:
            if not isinstance(raw_reference, str) or not raw_reference.strip():
                raise ValueError(f"{context}: explicit audio path must be a nonempty string.")
            file_name = PureWindowsPath(raw_reference).name
            source_identity: str | None = PureWindowsPath(raw_reference).as_posix()
        else:
            file_name = f"{_AUDIO_PREFIXES[side]}_{sample_id}_{label}.wav"
            source_identity = None

        if (
            not file_name
            or file_name in {".", ".."}
            or PureWindowsPath(file_name).name != file_name
            or Path(file_name).suffix.lower() != ".wav"
        ):
            raise ValueError(
                f"{context}: invalid WAV name derived from audio reference: {file_name!r}."
            )

        side_dir = self.split_root / f"{side}_audio"
        candidate = side_dir / file_name
        if not candidate.is_file():
            raise FileNotFoundError(f"{context}: missing 16 kHz WAV: {candidate}")
        resolved = candidate.resolve(strict=True)
        resolved_split_root = self.split_root.resolve(strict=True)
        resolved_side_dir = side_dir.resolve(strict=True)
        if (
            self.audio_root not in resolved.parents
            or resolved_split_root.parent != self.audio_root
            or resolved_split_root.name != f"{self.split}_audio"
            or resolved_split_root not in resolved.parents
            or resolved_side_dir.parent != resolved_split_root
            or resolved_side_dir not in resolved.parents
            or resolved_side_dir.name != f"{side}_audio"
        ):
            raise ValueError(f"{context}: WAV escapes its split/side directory: {candidate}")

        assignment_key = (side, resolved)
        previous = self._assignments.get(assignment_key)
        if previous is not None:
            previous_source, previous_text = previous
            if source_identity is not None and previous_source not in {None, source_identity}:
                raise ValueError(
                    f"{context}: distinct explicit audio references map to the same 16 kHz WAV: "
                    f"{previous_source!r} and {source_identity!r}."
                )
            if previous_text != record[side]:
                raise ValueError(
                    f"{context}: one 16 kHz WAV maps to different {side} texts; "
                    "duplicate ID/audio mapping is ambiguous."
                )
            if previous_source is None and source_identity is not None:
                self._assignments[assignment_key] = (source_identity, previous_text)
        else:
            self._assignments[assignment_key] = (source_identity, record[side])

        if resolved not in self._validated:
            try:
                info = sf.info(resolved)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"{context}: cannot read WAV metadata: {resolved}: {exc}") from exc
            if (
                info.format != "WAV"
                or info.channels != 1
                or info.frames < 1
                or info.samplerate != SAMPLE_RATE
            ):
                raise ValueError(
                    f"{context}: expected a nonempty mono {SAMPLE_RATE} Hz WAV, got "
                    f"format={info.format}, channels={info.channels}, "
                    f"frames={info.frames}, sample_rate={info.samplerate}: {resolved}"
                )
            self._validated.add(resolved)
        return str(resolved)
