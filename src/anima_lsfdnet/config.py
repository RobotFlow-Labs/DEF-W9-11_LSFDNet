from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class DataConfig:
    swir_dir: str = ""
    lwir_dir: str = ""
    label_dir: str = ""
    image_height: int = 512
    image_width: int = 640


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 8
    patch_size: int = 8
    attn_heads: int = 8


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 3407
    batch_size: int = 8
    lr: float = 1e-4
    total_iter: int = 30000
    warmup_iter: int = 500
    alpha: float = 0.5
    beta: float = 0.5
    sigma: float = 0.2
    gamma: float = 2.7


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    num_workers: int = 0
    deterministic: bool = True


@dataclass(frozen=True)
class LSFDNetConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeConfig


def _merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_config(raw: dict[str, Any]) -> LSFDNetConfig:
    data = DataConfig(**raw.get("data", {}))
    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**raw.get("train", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    return LSFDNetConfig(data=data, model=model, train=train, runtime=runtime)


def load_config(config_path: str | Path, fallback_path: str | Path | None = None) -> LSFDNetConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("rb") as fh:
        cfg_raw = tomllib.load(fh)

    if fallback_path is not None:
        fallback_path = Path(fallback_path)
        with fallback_path.open("rb") as fh:
            base_raw = tomllib.load(fh)
        merged = _merge(base_raw, cfg_raw)
    else:
        merged = cfg_raw

    return _to_config(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate LSFDNet config")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--fallback", default=None, help="Optional base TOML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.fallback)
    print(cfg)


if __name__ == "__main__":
    main()
