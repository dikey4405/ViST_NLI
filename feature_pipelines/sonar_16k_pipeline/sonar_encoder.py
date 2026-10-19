from __future__ import annotations

from typing import Any

import torch

from .config import FullSonar16kConfig


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu or auto.")
    return torch.device(value)


def load_text_pipeline(config: FullSonar16kConfig, device: torch.device) -> Any:
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    except ImportError as exc:
        raise ImportError("SONAR text pipeline is unavailable; install sonar-space.") from exc
    return TextToEmbeddingModelPipeline(
        encoder=config.text_encoder,
        tokenizer=config.text_tokenizer,
        device=device,
    )


def load_speech_pipeline(config: FullSonar16kConfig, device: torch.device) -> Any:
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    except ImportError as exc:
        raise ImportError("SONAR speech pipeline is unavailable; install sonar-space.") from exc
    return SpeechToEmbeddingModelPipeline(encoder=config.speech_encoder, device=device)


class FullSonarEncoder:
    """Encode fresh raw text and 16 kHz speech into CPU float32 vectors."""

    def __init__(self, config: FullSonar16kConfig) -> None:
        self.device = resolve_device(config.device)
        self.source_lang = config.text_source_lang
        self.batch_size = config.encoder_batch_size
        self.text_pipeline = load_text_pipeline(config, self.device)
        self.speech_pipeline = load_speech_pipeline(config, self.device)

    @staticmethod
    def _as_embeddings(values: Any, expected_rows: int, kind: str) -> torch.Tensor:
        tensor = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float32)
        if tensor.shape != (expected_rows, 1024) or not torch.isfinite(tensor).all():
            raise ValueError(
                f"{kind} encoder must return finite [{expected_rows}, 1024] embeddings; "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            result = self.text_pipeline.predict(
                texts,
                source_lang=self.source_lang,
                batch_size=self.batch_size,
            )
        return self._as_embeddings(result, len(texts), "Text")

    def encode_speech(self, audio_paths: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            result = self.speech_pipeline.predict(audio_paths, batch_size=self.batch_size)
        return self._as_embeddings(result, len(audio_paths), "Speech")
