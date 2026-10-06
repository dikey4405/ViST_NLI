from __future__ import annotations

import json
from pathlib import Path

from .utils import atomic_json_save


class VietnameseTextPreprocessor:
    """Optional cached Vietnamese word segmentation for PhoBERT input."""

    def __init__(
        self,
        *,
        use_word_segmentation: bool,
        cache_segmented_text: bool,
        cache_path: str | Path,
    ) -> None:
        self.use_word_segmentation = use_word_segmentation
        self.cache_segmented_text = cache_segmented_text
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str] = {}
        if cache_segmented_text and self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError(f"Segmented text cache must be a JSON object: {self.cache_path}")
            self.cache = {str(key): str(value) for key, value in payload.items()}

    def preprocess_many(self, texts: list[str]) -> list[str]:
        processed = [self.preprocess(text) for text in texts]
        if self.cache_segmented_text:
            atomic_json_save(self.cache, self.cache_path)
        return processed

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text input must be a non-empty string.")
        if not self.use_word_segmentation or _looks_word_segmented(text):
            return text
        if text in self.cache:
            return self.cache[text]
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError(
                "Vietnamese word segmentation is enabled but underthesea is not installed."
            ) from exc
        segmented = str(word_tokenize(text, format="text"))
        if not segmented.strip():
            raise ValueError("Word segmentation returned an empty string.")
        if self.cache_segmented_text:
            self.cache[text] = segmented
        return segmented


def _looks_word_segmented(text: str) -> bool:
    return any("_" in token for token in text.split())
