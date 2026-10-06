from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ModalityType(str, Enum):
    """Supported input modalities for each side of an NLI pair."""

    TEXT = "text"
    SPEECH = "speech"


class InputMode(str, Enum):
    """Supported premise-hypothesis modality combinations."""

    TEXT_TEXT = "text_text"
    TEXT_SPEECH = "text_speech"
    SPEECH_TEXT = "speech_text"
    SPEECH_SPEECH = "speech_speech"

    @property
    def premise_modality(self) -> ModalityType:
        return _INPUT_MODE_MODALITIES[self][0]

    @property
    def hypothesis_modality(self) -> ModalityType:
        return _INPUT_MODE_MODALITIES[self][1]


_INPUT_MODE_MODALITIES: dict[InputMode, tuple[ModalityType, ModalityType]] = {
    InputMode.TEXT_TEXT: (ModalityType.TEXT, ModalityType.TEXT),
    InputMode.TEXT_SPEECH: (ModalityType.TEXT, ModalityType.SPEECH),
    InputMode.SPEECH_TEXT: (ModalityType.SPEECH, ModalityType.TEXT),
    InputMode.SPEECH_SPEECH: (ModalityType.SPEECH, ModalityType.SPEECH),
}

INPUT_MODE_NAMES = tuple(mode.value for mode in InputMode)


RawInputValue = str | Path


@dataclass(frozen=True)
class NLIInputPair:
    """Raw premise-hypothesis pair before labels or sample metadata are attached."""

    premise: RawInputValue
    hypothesis: RawInputValue
    premise_modality: ModalityType
    hypothesis_modality: ModalityType
    input_mode: InputMode

    def to_dict(self) -> dict[str, Any]:
        return {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "premise_modality": self.premise_modality.value,
            "hypothesis_modality": self.hypothesis_modality.value,
            "input_mode": self.input_mode.value,
        }


@dataclass(frozen=True)
class NLISample:
    """A complete NLI sample returned by the dataset."""

    premise: RawInputValue
    hypothesis: RawInputValue
    premise_modality: ModalityType
    hypothesis_modality: ModalityType
    input_mode: InputMode
    label: str
    sample_index: int
    input_pairs: dict[str, dict[str, Any]]
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "id": self.id,
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "premise_modality": self.premise_modality.value,
            "hypothesis_modality": self.hypothesis_modality.value,
            "input_mode": self.input_mode.value,
            "input_pairs": self.input_pairs,
            "label": self.label,
        }
