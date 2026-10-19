from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "dev", "test")
SAMPLE_RATE = 16_000


def resolve_project_path(value: str | Path) -> Path:
    """Resolve CLI paths relative to KLTN, including a leading KLTN/ prefix."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0].lower() == PROJECT_ROOT.name.lower():
        return (PROJECT_ROOT.parent / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


@dataclass(frozen=True)
class FullSonar16kConfig:
    raw_data_dir: Path = PROJECT_ROOT / "Dataset"
    audio_root_16k: Path = PROJECT_ROOT / "Dataset_resampled_16k"
    output_dir: Path = PROJECT_ROOT / "Features" / "sonar_full_16k"
    splits: tuple[str, ...] = SPLITS
    batch_size: int = 8
    encoder_batch_size: int = 8
    device: str = "auto"
    text_encoder: str = "text_sonar_basic_encoder"
    text_tokenizer: str = "text_sonar_basic_encoder"
    speech_encoder: str = "sonar_speech_encoder_vie"
    text_source_lang: str = "vie_Latn"

    def __post_init__(self) -> None:
        for field_name in ("raw_data_dir", "audio_root_16k", "output_dir"):
            object.__setattr__(self, field_name, resolve_project_path(getattr(self, field_name)))
        if not self.splits or len(self.splits) != len(set(self.splits)):
            raise ValueError("splits must contain distinct train/dev/test names.")
        unknown = set(self.splits) - set(SPLITS)
        if unknown:
            raise ValueError(f"Unsupported splits: {sorted(unknown)}. Expected {SPLITS}.")
        if self.batch_size < 1 or self.encoder_batch_size < 1:
            raise ValueError("batch_size and encoder_batch_size must be positive.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")

        protected = (
            self.raw_data_dir,
            self.audio_root_16k,
            PROJECT_ROOT / "Features" / "sonar",
            PROJECT_ROOT / "Features" / "sonar_16k",
        )
        for root in protected:
            if _overlaps(self.output_dir, root.resolve()):
                raise ValueError(
                    f"Output directory must be separate from {root}: {self.output_dir}"
                )

    def split_path(self, split: str) -> Path:
        if split not in self.splits:
            raise ValueError(f"Split {split!r} was not selected.")
        candidates = [self.raw_data_dir / f"{split}{suffix}" for suffix in (".json", ".jsonl")]
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected one JSON/JSONL file for split={split} in {self.raw_data_dir}; "
                f"found {found}."
            )
        return found[0]
