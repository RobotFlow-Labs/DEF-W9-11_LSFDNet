from pathlib import Path

import numpy as np
from PIL import Image

from anima_lsfdnet.infer import run_folder_inference


def _write_img(path: Path, value: int) -> None:
    arr = np.full((32, 32), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_folder_inference(tmp_path: Path) -> None:
    sw = tmp_path / "SWIR"
    lw = tmp_path / "LWIR"
    out = tmp_path / "out"
    sw.mkdir(); lw.mkdir()

    _write_img(sw / "a.png", 100)
    _write_img(lw / "a.png", 140)

    records = run_folder_inference(sw, lw, out, checkpoint=None, device="cpu")
    assert len(records) == 1
    assert (out / "fused_images" / "a.png").exists()
    assert (out / "predictions.jsonl").exists()
