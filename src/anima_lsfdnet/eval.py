from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .io import load_gray_image, pair_files_by_stem
from .metrics import FusionMetrics, compute_fusion_metrics

PAPER_BASELINE = {
    "mAP50": 0.962,
    "mAP50_95": 0.770,
    "EN": 7.181,
    "SF": 21.022,
    "Qabf": 0.520,
}


def _mean_metrics(items: list[FusionMetrics]) -> dict[str, float]:
    if not items:
        return {"en": 0.0, "sf": 0.0, "sd": 0.0, "scd": 0.0, "qabf": 0.0}
    arr = {k: np.array([getattr(it, k) for it in items], dtype=np.float64) for k in ("en", "sf", "sd", "scd", "qabf")}
    return {k: float(v.mean()) for k, v in arr.items()}


def evaluate_fusion(swir_dir: str | Path, lwir_dir: str | Path, fused_dir: str | Path) -> dict[str, float]:
    fused_dir = Path(fused_dir)
    all_metrics: list[FusionMetrics] = []
    for swir_path, lwir_path in pair_files_by_stem(swir_dir, lwir_dir):
        f_path = fused_dir / f"{swir_path.stem}.png"
        if not f_path.exists():
            continue
        swir = load_gray_image(swir_path)
        lwir = load_gray_image(lwir_path)
        fused = load_gray_image(f_path)
        all_metrics.append(compute_fusion_metrics(fused, swir, lwir))
    return _mean_metrics(all_metrics)


def write_report(metrics: dict[str, float], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "fusion": metrics,
        "paper_baseline": PAPER_BASELINE,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))

    report = [
        "# LSFDNet Evaluation Report",
        "",
        "## Fusion Metrics",
        f"- EN: {metrics['en']:.4f} (paper: {PAPER_BASELINE['EN']})",
        f"- SF: {metrics['sf']:.4f} (paper: {PAPER_BASELINE['SF']})",
        f"- SD: {metrics['sd']:.4f}",
        f"- SCD: {metrics['scd']:.4f}",
        f"- Qabf: {metrics['qabf']:.4f} (paper: {PAPER_BASELINE['Qabf']})",
        "",
        "## Notes",
        "- Detection metrics require detector predictions and are tracked separately.",
    ]
    (out_dir / "eval_report.md").write_text("\n".join(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSFDNet fusion outputs")
    parser.add_argument("--swir-dir", required=True)
    parser.add_argument("--lwir-dir", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--out", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_fusion(args.swir_dir, args.lwir_dir, args.fused_dir)
    write_report(metrics, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
