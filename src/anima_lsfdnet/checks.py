from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LSFDNet config and required paths")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fallback", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.fallback)
    print("Loaded config OK")
    print(cfg)
    if cfg.data.swir_dir:
        print(f"SWIR path exists: {Path(cfg.data.swir_dir).exists()}")
    if cfg.data.lwir_dir:
        print(f"LWIR path exists: {Path(cfg.data.lwir_dir).exists()}")


if __name__ == "__main__":
    main()
