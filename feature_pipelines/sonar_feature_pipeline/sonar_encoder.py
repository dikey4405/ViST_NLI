from __future__ import annotations

from typing import Any

import torch

from .utils import validate_embedding_shape


class SonarEncoder:
    """Thin wrapper around SONAR text and speech encoders."""

    def __init__(
        self,
        *,
        text_encoder_name_or_path: str,
        text_tokenizer_name_or_path: str,
        speech_encoder_name_or_path: str,
        text_source_lang: str = "vie_Latn",
        batch_size: int = 8,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.text_source_lang = text_source_lang
        self.batch_size = batch_size
        self.text_pipeline = self._load_text_pipeline(
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_tokenizer_name_or_path=text_tokenizer_name_or_path,
        )
        self.speech_pipeline = self._load_speech_pipeline(
            speech_encoder_name_or_path=speech_encoder_name_or_path,
        )

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text inputs into a [batch_size, 1024] tensor."""

        embeddings = self.text_pipeline.predict(
            texts,
            source_lang=self.text_source_lang,
            batch_size=self.batch_size,
        )
        tensor = self._to_tensor(embeddings)
        validate_embedding_shape(tensor, name="text embedding")
        return tensor

    def encode_speech(self, audio_paths: list[str]) -> torch.Tensor:
        """Encode speech inputs into a [batch_size, 1024] tensor."""

        embeddings = self.speech_pipeline.predict(
            audio_paths,
            batch_size=self.batch_size,
        )
        tensor = self._to_tensor(embeddings)
        validate_embedding_shape(tensor, name="speech embedding")
        return tensor

    def _load_text_pipeline(
        self,
        *,
        text_encoder_name_or_path: str,
        text_tokenizer_name_or_path: str,
    ) -> Any:
        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        except ImportError as exc:
            raise ImportError(
                "SONAR text encoder is not available. Install SONAR and its dependencies "
                "before running feature extraction."
            ) from exc

        return TextToEmbeddingModelPipeline(
            encoder=text_encoder_name_or_path,
            tokenizer=text_tokenizer_name_or_path,
            device=self.device,
        )

    def _load_speech_pipeline(self, *, speech_encoder_name_or_path: str) -> Any:
        try:
            from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        except ImportError as exc:
            raise ImportError(
                "SONAR speech encoder is not available. Install SONAR and its dependencies "
                "before running feature extraction."
            ) from exc

        return SpeechToEmbeddingModelPipeline(
            encoder=speech_encoder_name_or_path,
            device=self.device,
        )

    def _to_tensor(self, embeddings: Any) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            tensor = embeddings
        else:
            tensor = torch.as_tensor(embeddings)
        return tensor.to(self.device, dtype=torch.float32)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
