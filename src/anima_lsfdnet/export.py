from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import create_model

ARTIFACTS = "/mnt/artifacts-datai"
PROJECT = "lsfdnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LSFDNet model artifacts")
    parser.add_argument("--checkpoint", required=False, default=None)
    parser.add_argument("--out", default=f"{ARTIFACTS}/exports/{PROJECT}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    return parser.parse_args()


def export_model(
    checkpoint: str | None, out_dir: str | Path, device: str, height: int, width: int
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(device=device)
    model.eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    # 1. PyTorch state dict
    torch.save(model.state_dict(), out_dir / "model.pth")
    print(f"[EXPORT] pth -> {out_dir / 'model.pth'}")

    # 2. Safetensors
    try:
        from safetensors.torch import save_file

        save_file(model.state_dict(), out_dir / "model.safetensors")
        print(f"[EXPORT] safetensors -> {out_dir / 'model.safetensors'}")
    except ImportError:
        print("[WARN] safetensors not installed, skipping")

    # 3. ONNX
    sw = torch.randn(1, 1, height, width, device=device)
    lw = torch.randn(1, 1, height, width, device=device)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        (sw, lw),
        onnx_path,
        input_names=["swir", "lwir"],
        output_names=["fused", "fused_feature"],
        opset_version=17,
        dynamic_axes={
            "swir": {0: "batch", 2: "height", 3: "width"},
            "lwir": {0: "batch", 2: "height", 3: "width"},
            "fused": {0: "batch", 2: "height", 3: "width"},
            "fused_feature": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"[EXPORT] ONNX -> {onnx_path}")

    # 4. TensorRT (if available)
    try:
        import shutil
        import subprocess

        trt_toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
        if trt_toolkit.exists() and shutil.which("trtexec"):
            for precision in ("fp16", "fp32"):
                trt_path = out_dir / f"model_{precision}.engine"
                cmd = [
                    "python",
                    str(trt_toolkit),
                    "--onnx",
                    str(onnx_path),
                    "--output",
                    str(trt_path),
                    "--precision",
                    precision,
                ]
                print(f"[EXPORT] TRT {precision} -> {trt_path}")
                subprocess.run(cmd, check=True)
        elif shutil.which("trtexec"):
            for precision in ("fp16", "fp32"):
                trt_path = out_dir / f"model_{precision}.engine"
                flag = "--fp16" if precision == "fp16" else ""
                cmd = f"trtexec --onnx={onnx_path} --saveEngine={trt_path} {flag}".strip()
                print(f"[EXPORT] TRT {precision} via trtexec -> {trt_path}")
                subprocess.run(cmd.split(), check=True)
        else:
            print("[WARN] TensorRT not available, skipping TRT export")
    except Exception as e:
        print(f"[WARN] TRT export failed: {e}")

    print(f"[DONE] All exports in {out_dir}")


def main() -> None:
    args = parse_args()
    export_model(args.checkpoint, args.out, args.device, args.height, args.width)


if __name__ == "__main__":
    main()
