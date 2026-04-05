from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .io import load_gray_image, pair_files_by_stem, save_gray_image
from .model import create_model


def _to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)


def run_folder_inference(
    swir_dir: str | Path,
    lwir_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = out_dir / "fused_images"
    fused_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(device=device)
    model.eval()

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    records: list[dict[str, object]] = []
    with torch.no_grad():
        for swir_path, lwir_path in pair_files_by_stem(swir_dir, lwir_dir):
            swir = load_gray_image(swir_path)
            lwir = load_gray_image(lwir_path)
            sw_t = _to_tensor(swir, device)
            lw_t = _to_tensor(lwir, device)
            t0 = time.perf_counter()
            fused_t, _ = model(sw_t, lw_t)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            fused = fused_t.squeeze().detach().cpu().numpy()
            out_path = fused_dir / f"{swir_path.stem}.png"
            save_gray_image(out_path, fused)
            records.append({"id": swir_path.stem, "output": str(out_path), "runtime_ms": dt_ms, "shape": list(fused.shape)})

    (out_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSFDNet fusion inference on SWIR/LWIR folders")
    parser.add_argument("--swir-dir", required=True)
    parser.add_argument("--lwir-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_folder_inference(
        swir_dir=args.swir_dir,
        lwir_dir=args.lwir_dir,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(f"Processed {len(results)} paired samples")


if __name__ == "__main__":
    main()
