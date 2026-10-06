from __future__ import annotations

from torch.utils.data import DataLoader

from ...data.config import DataLoaderConfig, DatasetConfig
from ...data.datamodule import build_dataloader as build_data_dataloader
from ...data.schemas import InputMode
from .config import FeatureExtractionConfig


def build_dataloader(config: FeatureExtractionConfig, split: str) -> DataLoader:
    """Build a DataLoader by delegating data handling to source.data."""

    data_path = config.get_split_path(split)
    dataset_config = DatasetConfig(
        data_path=data_path,
        input_mode=InputMode.TEXT_TEXT,
        label_mapping=config.label_mapping,
        validate_audio_exists=config.validate_audio_exists,
        allow_ambiguous_audio=config.allow_ambiguous_audio,
    )
    dataloader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    return build_data_dataloader(dataset_config, dataloader_config, shuffle=False)
