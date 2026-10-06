from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .pooling import masked_mean_pooling
from .schemas import AudioLoadResult, AudioOverlengthPolicy
from .utils import autocast_context, freeze_module, resolve_device


class AudioSkippedError(RuntimeError):
    """Controlled signal for an audio sample skipped by configuration."""


class AudioOverlengthError(ValueError):
    """Raised when an overlength audio file is configured to fail extraction."""


def load_audio(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
    max_audio_seconds: float,
    overlength_policy: AudioOverlengthPolicy,
) -> AudioLoadResult:
    """Load mono audio, resample it, and apply the configured duration policy."""

    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError("soundfile and scipy are required for XLS-R audio loading.") from exc

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform_array, original_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform_array.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    original_num_channels = int(waveform_array.shape[1])
    mono = waveform_array.mean(axis=1)
    was_resampled = original_sample_rate != target_sample_rate
    if was_resampled:
        divisor = math.gcd(int(original_sample_rate), int(target_sample_rate))
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            int(original_sample_rate) // divisor,
        ).astype(np.float32, copy=False)

    max_samples = int(round(max_audio_seconds * target_sample_rate))
    was_truncated = mono.shape[0] > max_samples
    if was_truncated:
        if overlength_policy == AudioOverlengthPolicy.RAISE:
            duration = mono.shape[0] / target_sample_rate
            raise AudioOverlengthError(
                f"Audio duration {duration:.2f}s exceeds configured limit "
                f"{max_audio_seconds:.2f}s: {path}"
            )
        if overlength_policy == AudioOverlengthPolicy.SKIP:
            raise AudioSkippedError(f"Audio exceeds duration limit and was skipped: {path}")
        mono = mono[:max_samples]

    return AudioLoadResult(
        waveform=torch.from_numpy(np.ascontiguousarray(mono)),
        sample_rate=target_sample_rate,
        original_sample_rate=int(original_sample_rate),
        original_num_channels=original_num_channels,
        was_resampled=was_resampled,
        was_truncated=was_truncated,
    )


class XLSRSpeechEncoder(nn.Module):
    """Frozen XLS-R encoder with frame-level padding-aware pooling."""

    def __init__(
        self,
        *,
        model_name: str,
        target_sample_rate: int,
        max_audio_seconds: float,
        overlength_policy: AudioOverlengthPolicy,
        expected_output_dim: int = 1024,
        freeze: bool = True,
        device: str = "auto",
        mixed_precision: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for XLS-R feature extraction.") from exc

        self.device = resolve_device(device)
        self.target_sample_rate = target_sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.overlength_policy = overlength_policy
        self.expected_output_dim = expected_output_dim
        self.freeze = freeze
        self.mixed_precision = mixed_precision
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        extractor_rate = int(getattr(self.feature_extractor, "sampling_rate", target_sample_rate))
        if extractor_rate != target_sample_rate:
            raise ValueError(
                f"Configured target_sample_rate={target_sample_rate} does not match "
                f"feature extractor rate {extractor_rate}."
            )
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        if freeze:
            freeze_module(self.model)

    def encode(self, audio_paths: list[str]) -> torch.Tensor:
        loaded = [
            load_audio(
                path,
                target_sample_rate=self.target_sample_rate,
                max_audio_seconds=self.max_audio_seconds,
                overlength_policy=self.overlength_policy,
            )
            for path in audio_paths
        ]
        return self.encode_waveforms([item.waveform for item in loaded])

    def encode_waveforms(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        if not waveforms:
            raise ValueError("waveforms must contain at least one item.")
        arrays = [waveform.detach().cpu().numpy() for waveform in waveforms]
        inputs = self.feature_extractor(
            arrays,
            sampling_rate=self.target_sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if "attention_mask" not in inputs:
            raise ValueError("XLS-R feature extractor did not return an attention mask.")

        gradient_context = torch.inference_mode() if self.freeze else nullcontext()
        mixed_precision_context = autocast_context(self.device, self.mixed_precision)
        with gradient_context, mixed_precision_context:
            outputs: Any = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            feature_mask = self._build_feature_attention_mask(
                hidden_states.shape[1], inputs["attention_mask"]
            )
            pooled = masked_mean_pooling(hidden_states, feature_mask)
        if pooled.ndim != 2 or pooled.shape[-1] != self.expected_output_dim:
            raise ValueError(
                f"XLS-R pooled output must have shape [B, {self.expected_output_dim}], "
                f"got {tuple(pooled.shape)}."
            )
        return pooled.float()

    def _build_feature_attention_mask(
        self,
        feature_length: int,
        waveform_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            mask = self.model._get_feature_vector_attention_mask(
                feature_length,
                waveform_attention_mask,
            )
        elif hasattr(self.model, "_get_feat_extract_output_lengths"):
            waveform_lengths = waveform_attention_mask.sum(dim=-1)
            feature_lengths = self.model._get_feat_extract_output_lengths(waveform_lengths)
            positions = torch.arange(feature_length, device=waveform_attention_mask.device)
            mask = positions.unsqueeze(0) < feature_lengths.unsqueeze(1)
        else:
            raise RuntimeError("XLS-R model cannot convert waveform masks to feature-frame masks.")
        if mask.shape != (waveform_attention_mask.shape[0], feature_length):
            raise ValueError(
                "Feature attention mask shape does not match hidden states: "
                f"{tuple(mask.shape)} versus {(waveform_attention_mask.shape[0], feature_length)}."
            )
        return mask
