from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .schemas import InputMode, ModalityType, NLIInputPair, NLISample
from .utils import (
    build_audio_index,
    get_label,
    parse_input_mode,
    read_json_or_jsonl,
    resolve_audio_path,
    resolve_audio_root,
    validate_raw_nli_record,
)


class NLIMultimodalDataset(Dataset):
    """Return raw text and/or WAV paths for multimodal Vietnamese NLI samples."""

    def __init__(
        self,
        data_path: str | Path,
        input_mode: InputMode | str = InputMode.TEXT_TEXT,
        *,
        audio_root: str | Path | None = None,
        label_mapping: dict[str, int] | None = None,
        validate_audio_exists: bool = True,
        allow_ambiguous_audio: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.input_mode = parse_input_mode(input_mode)
        self.audio_root = resolve_audio_root(self.data_path, audio_root)
        self.label_mapping = label_mapping
        self.validate_audio_exists = validate_audio_exists
        self.allow_ambiguous_audio = allow_ambiguous_audio
        self.records = read_json_or_jsonl(self.data_path)
        self.speech_sides = ["premise", "hypothesis"]
        self.audio_index = build_audio_index(self.audio_root, self.speech_sides) if self.speech_sides else {}

        for index, record in enumerate(self.records):
            validate_raw_nli_record(record, index)
            self._validate_label(record, index)
        self.integrity_report = self._build_integrity_report()
        self._validate_audio_ambiguity()
        self._validate_audio_files_exist()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one normalized sample without embedding, fusion, or label prediction."""

        record = self.records[index]
        sample_id = self._get_id(record)
        label = get_label(record)

        input_pairs = self._build_input_pairs(record=record, sample_id=sample_id, label=label)
        selected_pair = input_pairs[self.input_mode.value]

        return NLISample(
            id=sample_id,
            sample_index=index,
            premise=selected_pair["premise"],
            hypothesis=selected_pair["hypothesis"],
            premise_modality=self.input_mode.premise_modality,
            hypothesis_modality=self.input_mode.hypothesis_modality,
            input_mode=self.input_mode,
            input_pairs=input_pairs,
            label=label,
        ).to_dict()

    def _build_input_pairs(
        self,
        *,
        record: dict[str, Any],
        sample_id: str | None,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        text_premise = self._resolve_input_value(
            record=record,
            side="premise",
            modality=ModalityType.TEXT,
            sample_id=sample_id,
            label=label,
        )
        text_hypothesis = self._resolve_input_value(
            record=record,
            side="hypothesis",
            modality=ModalityType.TEXT,
            sample_id=sample_id,
            label=label,
        )
        speech_premise = self._resolve_input_value(
            record=record,
            side="premise",
            modality=ModalityType.SPEECH,
            sample_id=sample_id,
            label=label,
        )
        speech_hypothesis = self._resolve_input_value(
            record=record,
            side="hypothesis",
            modality=ModalityType.SPEECH,
            sample_id=sample_id,
            label=label,
        )

        pairs = {
            InputMode.TEXT_TEXT: (text_premise, text_hypothesis),
            InputMode.TEXT_SPEECH: (text_premise, speech_hypothesis),
            InputMode.SPEECH_TEXT: (speech_premise, text_hypothesis),
            InputMode.SPEECH_SPEECH: (speech_premise, speech_hypothesis),
        }
        return {
            mode.value: NLIInputPair(
                premise=premise,
                hypothesis=hypothesis,
                premise_modality=mode.premise_modality,
                hypothesis_modality=mode.hypothesis_modality,
                input_mode=mode,
            ).to_dict()
            for mode, (premise, hypothesis) in pairs.items()
        }

    def _resolve_input_value(
        self,
        *,
        record: dict[str, Any],
        side: str,
        modality: ModalityType,
        sample_id: str | None,
        label: str,
    ) -> Any:
        """Resolve one side of an NLI pair as either raw text or a WAV path."""

        if modality == ModalityType.TEXT:
            return record[side]

        if sample_id is None:
            raise ValueError(f"Cannot build {side} speech path because sample id is missing.")

        explicit_field = f"{side}_audio"
        if explicit_field in record and record[explicit_field]:
            path = Path(record[explicit_field])
            if not path.is_absolute():
                path = self.audio_root / path
            if self.validate_audio_exists and not path.exists():
                raise FileNotFoundError(f"Audio file from '{explicit_field}' does not exist: {path}")
            return str(path)

        return resolve_audio_path(self.audio_index, side, sample_id, label)

    def _validate_label(self, record: dict[str, Any], index: int) -> None:
        label = get_label(record)
        if self.label_mapping is not None and label not in self.label_mapping:
            valid = ", ".join(self.label_mapping)
            raise ValueError(f"Unknown label '{label}' at sample index {index}. Expected one of: {valid}")

    def _build_integrity_report(self) -> dict[str, int]:
        grouped: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            grouped[(self._get_id(record), get_label(record))].append(record)

        duplicate_groups = [group for group in grouped.values() if len(group) > 1]
        premise_varied = sum(1 for group in duplicate_groups if len({row["premise"] for row in group}) > 1)
        hypothesis_varied = sum(1 for group in duplicate_groups if len({row["hypothesis"] for row in group}) > 1)

        return {
            "num_samples": len(self.records),
            "num_unique_id_label_pairs": len(grouped),
            "num_duplicate_id_label_groups": len(duplicate_groups),
            "num_duplicate_groups_with_varied_premise": premise_varied,
            "num_duplicate_groups_with_varied_hypothesis": hypothesis_varied,
        }

    def _validate_audio_ambiguity(self) -> None:
        if self.allow_ambiguous_audio:
            return

        ambiguous_sides: list[str] = []
        if (
            "premise" in self.speech_sides
            and self.integrity_report["num_duplicate_groups_with_varied_premise"] > 0
        ):
            ambiguous_sides.append("premise")
        if (
            "hypothesis" in self.speech_sides
            and self.integrity_report["num_duplicate_groups_with_varied_hypothesis"] > 0
        ):
            ambiguous_sides.append("hypothesis")

        if ambiguous_sides:
            sides = ", ".join(ambiguous_sides)
            raise ValueError(
                "Ambiguous speech mapping detected for side(s): "
                f"{sides}. Some duplicated (id, label) groups contain different text "
                "but share the same derived WAV path. "
                "Set allow_ambiguous_audio=True to keep loading the raw data, or deduplicate/fix the audio filenames."
            )

    def _validate_audio_files_exist(self) -> None:
        if not self.validate_audio_exists:
            return
        if not self.speech_sides:
            return

        for index, record in enumerate(self.records):
            sample_id = self._get_id(record)
            if sample_id is None:
                raise ValueError(f"Cannot validate speech path because sample id is missing at index {index}.")
            label = get_label(record)

            for side in self.speech_sides:
                explicit_field = f"{side}_audio"
                if explicit_field in record and record[explicit_field]:
                    path = Path(record[explicit_field])
                    if not path.is_absolute():
                        path = self.audio_root / path
                    if not path.exists():
                        raise FileNotFoundError(f"Audio file from '{explicit_field}' does not exist: {path}")
                    continue

                resolve_audio_path(self.audio_index, side, sample_id, label)

    @staticmethod
    def _get_id(record: dict[str, Any]) -> str | None:
        value = record.get("id")
        return None if value is None else str(value)
