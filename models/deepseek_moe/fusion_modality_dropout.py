"""Backward-compatible imports for the shared fusion modality dropout."""

from ..common.fusion_modality_dropout import (
    EXPECTED_NUM_MODES,
    ST_INDEX,
    TT_INDEX,
    FusionModalityDropout,
    register_fusion_modality_dropout,
)

__all__ = [
    "EXPECTED_NUM_MODES",
    "ST_INDEX",
    "TT_INDEX",
    "FusionModalityDropout",
    "register_fusion_modality_dropout",
]
