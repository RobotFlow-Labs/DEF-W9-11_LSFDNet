from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pair_files_by_stem(swir_dir: str | Path, lwir_dir: str | Path) -> list[tuple[Path, Path]]:
    swir_dir = Path(swir_dir)
    lwir_dir = Path(lwir_dir)
    sw_map = {
        p.stem: p for p in swir_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }
    lw_map = {
        p.stem: p for p in lwir_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }
    stems = sorted(set(sw_map) & set(lw_map))
    return [(sw_map[s], lw_map[s]) for s in stems]


def load_gray_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_gray_image(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def to_base64_png(array: np.ndarray) -> str:
    arr = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
