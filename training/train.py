from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .trainer import run_training_from_config


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> argparse.Namespace:
    """Parse arguments for the shared saved-feature NLI training entry point."""

    parser = argparse.ArgumentParser(
        description="Train one NLI model from one saved multimodal feature source."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        required=default_config is None,
        help="NLI model YAML config path or config file name.",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default=None,
        help="Feature source key from feature_sources, such as sonar or phobert_xlsr.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> None:
    """Run the shared trainer without loading any feature encoder."""

    args = parse_args(argv, default_config=default_config)
    run_training_from_config(
        args.config,
        feature_source=args.feature_source,
    )


if __name__ == "__main__":
    main()
