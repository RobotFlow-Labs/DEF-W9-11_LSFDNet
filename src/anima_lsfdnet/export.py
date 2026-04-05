from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LSFDNet model artifacts")
    parser.add_argument("--checkpoint", required=False, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    return parser.parse_args()


def export_model(checkpoint: str | None, out_dir: str | Path, device: str, height: int, width: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(device=device)
    model.eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)

    torch.save(model.state_dict(), out_dir / "model.pth")

    sw = torch.randn(1, 1, height, width, device=device)
    lw = torch.randn(1, 1, height, width, device=device)
    torch.onnx.export(
        model,
        (sw, lw),
        out_dir / "model.onnx",
        input_names=["swir", "lwir"],
        output_names=["fused", "fused_feature"],
        opset_version=17,
    )


if __name__ == "__main__":
    args = parse_args()
    export_model(args.checkpoint, args.out, args.device, args.height, args.width)
