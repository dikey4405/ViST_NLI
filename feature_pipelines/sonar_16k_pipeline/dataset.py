from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ...data.config import DEFAULT_LABEL_MAPPING
from ...data.schemas import InputMode, ModalityType, NLIInputPair, NLISample
from ...data.utils import get_label, read_json_or_jsonl, validate_raw_nli_record
from .audio_resolver import AudioResolver


class FullSonar16kDataset(Dataset):
    """Read raw NLI records and preflight their 16 kHz audio references."""

    def __init__(self, data_path: Path, audio_root_16k: Path, split: str) -> None:
        self.data_path = Path(data_path)
        self.split = split
        self.resolver = AudioResolver(audio_root_16k, split)
        records = read_json_or_jsonl(self.data_path)
        if not records:
            raise ValueError(f"Split {split!r} contains no NLI samples: {self.data_path}")
        self.samples = [self._prepare_sample(record, index) for index, record in enumerate(records)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def _prepare_sample(self, record: dict[str, Any], index: int) -> dict[str, Any]:
        try:
            validate_raw_nli_record(record, index)
        except ValueError as exc:
            raise ValueError(f"split={self.split}: {exc}") from exc
        for side in ("premise", "hypothesis"):
            if not isinstance(record[side], str) or not record[side].strip():
                raise ValueError(
                    f"split={self.split}, sample_index={index}: {side} must be nonempty text."
                )
        sample_id = record.get("id")
        if sample_id is None or not str(sample_id).strip():
            raise ValueError(f"split={self.split}, sample_index={index}: missing sample ID.")
        sample_id = str(sample_id)
        try:
            label = get_label(record)
        except ValueError as exc:
            raise ValueError(
                f"split={self.split}, sample_index={index}, id={sample_id!r}: {exc}"
            ) from exc
        if label not in DEFAULT_LABEL_MAPPING:
            raise ValueError(
                f"split={self.split}, sample_index={index}, id={sample_id!r}: "
                f"unknown NLI label {label!r}."
            )

        text_premise = record["premise"]
        text_hypothesis = record["hypothesis"]
        speech_premise = self.resolver.resolve(
            record, sample_index=index, sample_id=sample_id, label=label, side="premise"
        )
        speech_hypothesis = self.resolver.resolve(
            record, sample_index=index, sample_id=sample_id, label=label, side="hypothesis"
        )
        values = {
            InputMode.TEXT_TEXT: (text_premise, text_hypothesis),
            InputMode.TEXT_SPEECH: (text_premise, speech_hypothesis),
            InputMode.SPEECH_TEXT: (speech_premise, text_hypothesis),
            InputMode.SPEECH_SPEECH: (speech_premise, speech_hypothesis),
        }
        input_pairs = {
            mode.value: NLIInputPair(
                premise=premise,
                hypothesis=hypothesis,
                premise_modality=mode.premise_modality,
                hypothesis_modality=mode.hypothesis_modality,
                input_mode=mode,
            ).to_dict()
            for mode, (premise, hypothesis) in values.items()
        }
        return NLISample(
            id=sample_id,
            sample_index=index,
            premise=text_premise,
            hypothesis=text_hypothesis,
            premise_modality=ModalityType.TEXT,
            hypothesis_modality=ModalityType.TEXT,
            input_mode=InputMode.TEXT_TEXT,
            input_pairs=input_pairs,
            label=label,
        ).to_dict()
