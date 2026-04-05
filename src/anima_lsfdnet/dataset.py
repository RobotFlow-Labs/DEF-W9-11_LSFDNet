from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairSample:
    swir: Path
    lwir: Path
    label: Path | None


def _index_images(folder: Path) -> dict[str, Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return {p.stem: p for p in files}


def pair_samples(swir_dir: Path, lwir_dir: Path, label_dir: Path | None = None) -> list[PairSample]:
    sw_map = _index_images(swir_dir)
    lw_map = _index_images(lwir_dir)
    stems = sorted(set(sw_map) & set(lw_map))
    out: list[PairSample] = []
    for stem in stems:
        label = (label_dir / f"{stem}.txt") if label_dir is not None else None
        if label is not None and not label.exists():
            label = None
        out.append(PairSample(swir=sw_map[stem], lwir=lw_map[stem], label=label))
    return out


def _load_gray(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_yolo_label(path: Path | None) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(v) for v in line.split()[:5]])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class NSLSRDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, swir_dir: str | Path, lwir_dir: str | Path, label_dir: str | Path | None = None
    ):
        self.swir_dir = Path(swir_dir)
        self.lwir_dir = Path(lwir_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.samples = pair_samples(self.swir_dir, self.lwir_dir, self.label_dir)
        if not self.samples:
            raise ValueError(f"No paired samples found in {self.swir_dir} and {self.lwir_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        swir = _load_gray(sample.swir)
        lwir = _load_gray(sample.lwir)
        labels = _load_yolo_label(sample.label)
        return {"swir": swir, "lwir": lwir, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NSLSR-style paired dataset")
    parser.add_argument("--swir-dir", required=True)
    parser.add_argument("--lwir-dir", required=True)
    parser.add_argument("--label-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = NSLSRDataset(args.swir_dir, args.lwir_dir, args.label_dir)
    first = ds[0]
    print(
        {
            "num_samples": len(ds),
            "swir_shape": tuple(first["swir"].shape),
            "lwir_shape": tuple(first["lwir"].shape),
            "labels_shape": tuple(first["labels"].shape),
        }
    )


if __name__ == "__main__":
    main()
