from __future__ import annotations

from pathlib import Path

from .train import main


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "conventional_moe.yaml"


if __name__ == "__main__":
    main(default_config=CONFIG_PATH)
