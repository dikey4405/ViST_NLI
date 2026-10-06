from __future__ import annotations

from typing import Any

from ...data.schemas import INPUT_MODE_NAMES
from .schemas import InputMode, ModalityType


def build_input_pairs(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the four input pairs already normalized by source.data."""

    if "input_pairs" not in sample:
        raise ValueError(
            "Sample must contain 'input_pairs'. "
            "Use KLTN.source.data.NLIMultimodalDataset first."
        )
    _validate_input_pairs(sample["input_pairs"])
    return sample["input_pairs"]


def build_batch_input_pairs(batch: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """Return the four input-mode batches produced by source.data.NLICollator."""

    if "input_pairs" not in batch:
        raise ValueError(
            "Batch must contain 'input_pairs'. Use KLTN.source.data.NLICollator first."
        )
    _validate_batch_input_pairs(batch["input_pairs"])
    return batch["input_pairs"]


def get_pair_modalities(input_mode: str | InputMode) -> tuple[ModalityType, ModalityType]:
    mode = input_mode if isinstance(input_mode, InputMode) else InputMode(input_mode)
    return mode.premise_modality, mode.hypothesis_modality


def _validate_input_pairs(input_pairs: dict[str, dict[str, Any]]) -> None:
    _validate_required_modes(input_pairs, context="Input pairs")

    for mode in InputMode:
        pair = input_pairs[mode.value]
        _validate_pair_modalities(
            mode=mode,
            premise_modality=pair.get("premise_modality"),
            hypothesis_modality=pair.get("hypothesis_modality"),
        )


def _validate_batch_input_pairs(input_pairs: dict[str, dict[str, list[Any]]]) -> None:
    _validate_required_modes(input_pairs, context="Batch input pairs")

    for mode in InputMode:
        pair_batch = input_pairs[mode.value]
        required_keys = {
            "premise",
            "hypothesis",
            "premise_modality",
            "hypothesis_modality",
            "input_mode",
        }
        missing_keys = required_keys - set(pair_batch)
        if missing_keys:
            raise ValueError(
                f"Batch input pair '{mode.value}' is missing key(s): "
                f"{', '.join(sorted(missing_keys))}"
            )
        if not pair_batch["premise"]:
            raise ValueError(f"Batch input pair '{mode.value}' is empty.")
        if len(pair_batch["premise"]) != len(pair_batch["hypothesis"]):
            raise ValueError(
                f"Batch input pair '{mode.value}' has different premise/hypothesis lengths: "
                f"{len(pair_batch['premise'])} and {len(pair_batch['hypothesis'])}."
            )
        _validate_batch_values(
            values=pair_batch["premise_modality"],
            expected=mode.premise_modality.value,
            field_name="premise_modality",
            mode=mode,
        )
        _validate_batch_values(
            values=pair_batch["hypothesis_modality"],
            expected=mode.hypothesis_modality.value,
            field_name="hypothesis_modality",
            mode=mode,
        )
        _validate_batch_values(
            values=pair_batch["input_mode"],
            expected=mode.value,
            field_name="input_mode",
            mode=mode,
        )
        _validate_pair_modalities(
            mode=mode,
            premise_modality=pair_batch["premise_modality"][0],
            hypothesis_modality=pair_batch["hypothesis_modality"][0],
        )


def _validate_required_modes(input_pairs: dict[str, Any], *, context: str) -> None:
    missing = set(INPUT_MODE_NAMES) - set(input_pairs)
    if missing:
        raise ValueError(f"{context} are missing mode(s): {', '.join(sorted(missing))}")


def _validate_pair_modalities(
    *,
    mode: InputMode,
    premise_modality: Any,
    hypothesis_modality: Any,
) -> None:
    if premise_modality != mode.premise_modality.value:
        raise ValueError(
            f"Mode '{mode.value}' expected premise modality "
            f"'{mode.premise_modality.value}', got '{premise_modality}'."
        )
    if hypothesis_modality != mode.hypothesis_modality.value:
        raise ValueError(
            f"Mode '{mode.value}' expected hypothesis modality "
            f"'{mode.hypothesis_modality.value}', got '{hypothesis_modality}'."
        )


def _validate_batch_values(
    *,
    values: list[Any],
    expected: str,
    field_name: str,
    mode: InputMode,
) -> None:
    if not values:
        raise ValueError(f"Batch input pair '{mode.value}' has empty '{field_name}'.")

    invalid_indices = [index for index, value in enumerate(values) if value != expected]
    if invalid_indices:
        first_index = invalid_indices[0]
        raise ValueError(
            f"Batch input pair '{mode.value}' expected every '{field_name}' to be '{expected}', "
            f"but index {first_index} has '{values[first_index]}'."
        )
