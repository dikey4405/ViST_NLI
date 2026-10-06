from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .schemas import InputMode


REQUIRED_NLI_FIELDS = ("premise", "hypothesis")
DEFAULT_LABEL_FIELDS = ("label", "gold_label")
DEFAULT_LABELS = ("entailment", "contradiction", "neutral")
AUDIO_SIDE_CONFIG: dict[str, tuple[str, str]] = {
    "premise": ("Premise", "prem"),
    "hypothesis": ("Hypothesis", "hypo"),
}
AudioIndex = dict[tuple[str, str, str], str]


def resolve_audio_root(data_path: str | Path, audio_root: str | Path | None = None) -> Path:
    """Locate a split's audio root, supporting current and legacy layouts."""

    if audio_root is not None:
        return Path(audio_root)
    path = Path(data_path)
    split_audio = path.parent / f"{path.stem}_audio"
    return split_audio if split_audio.is_dir() else path.parent


def resolve_audio_directory(audio_root: str | Path, side: str) -> Path:
    """Resolve one audio side without guessing individual WAV filenames."""

    legacy_folder, _ = _get_audio_side_config(side)
    root = Path(audio_root)
    candidates = (root / f"{side}_audio", root / legacy_folder)
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous {side} audio directories: {existing}")
    if not existing:
        raise FileNotFoundError(f"Missing {side} audio directory. Checked: {candidates}")
    return existing[0]


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read NLI samples from a JSON list, a {'data': [...]} JSON object, or JSONL."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(data_path)
    if suffix == ".json":
        return _read_json(data_path)
    raise ValueError(f"Unsupported data file extension '{data_path.suffix}'. Expected .json or .jsonl.")


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise ValueError(f"JSON file must contain a list of samples or a dict with a 'data' list: {path}")

    return _ensure_record_list(records, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            records.append(record)
    return records


def _ensure_record_list(records: Iterable[Any], path: Path) -> list[dict[str, Any]]:
    normalized = list(records)
    for index, record in enumerate(normalized):
        if not isinstance(record, dict):
            raise ValueError(f"Sample at index {index} in {path} must be an object.")
    return normalized


def parse_input_mode(input_mode: InputMode | str) -> InputMode:
    """Normalize a string or enum input mode to InputMode."""

    if isinstance(input_mode, InputMode):
        return input_mode
    try:
        return InputMode(input_mode)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in InputMode)
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Expected one of: {valid}") from exc


def get_label(record: dict[str, Any], label_fields: tuple[str, ...] = DEFAULT_LABEL_FIELDS) -> str:
    """Return the NLI label from a raw record."""

    for field in label_fields:
        value = record.get(field)
        if value is not None:
            return str(value)
    expected = " or ".join(label_fields)
    raise ValueError(f"Missing label field. Expected {expected}.")


def validate_raw_nli_record(record: dict[str, Any], index: int) -> None:
    """Validate required raw text fields before a record is converted to a sample."""

    missing = [field for field in REQUIRED_NLI_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Sample at index {index} is missing required field(s): {', '.join(missing)}")


def build_audio_index(
    split_dir: str | Path,
    sides: Iterable[str] = ("premise", "hypothesis"),
    labels: Iterable[str] = DEFAULT_LABELS,
) -> AudioIndex:
    """Index existing WAV files as (side, sample_id, label) -> path."""

    audio_index: AudioIndex = {}
    label_set = set(labels)

    for side in sides:
        audio_dir = resolve_audio_directory(split_dir, side)

        for wav_path in sorted(audio_dir.glob("*.wav")):
            sample_id, label = parse_audio_filename(side, wav_path.name, label_set)
            key = (side, sample_id, label)
            if key in audio_index:
                raise ValueError(f"Duplicate audio file for {key}: {audio_index[key]} and {wav_path}")
            audio_index[key] = str(wav_path)

    return audio_index


def parse_audio_filename(side: str, file_name: str, labels: set[str] | None = None) -> tuple[str, str]:
    """Parse a WAV file name into sample_id and label."""

    _, prefix = _get_audio_side_config(side)
    label_set = labels or set(DEFAULT_LABELS)
    label_pattern = "|".join(re.escape(label) for label in sorted(label_set))
    pattern = rf"^{re.escape(prefix)}_(.+)_({label_pattern})\.wav$"
    match = re.match(pattern, file_name)
    if match is None:
        expected = f"{prefix}_{{id}}_{{label}}.wav"
        raise ValueError(f"Invalid {side} audio filename '{file_name}'. Expected format: {expected}")
    return match.group(1), match.group(2)


def resolve_audio_path(audio_index: AudioIndex, side: str, sample_id: str, label: str) -> str:
    """Look up an audio path from an AudioIndex."""

    key = (side, sample_id, label)
    try:
        return audio_index[key]
    except KeyError as exc:
        raise FileNotFoundError(
            f"No {side} audio file found for sample_id='{sample_id}', label='{label}'."
        ) from exc


def _get_audio_side_config(side: str) -> tuple[str, str]:
    if side not in {"premise", "hypothesis"}:
        raise ValueError(f"Audio side must be 'premise' or 'hypothesis', got '{side}'.")
    return AUDIO_SIDE_CONFIG[side]
