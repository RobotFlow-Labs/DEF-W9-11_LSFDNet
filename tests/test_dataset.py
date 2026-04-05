from pathlib import Path

import numpy as np
from PIL import Image

from anima_lsfdnet.dataset import NSLSRDataset, pair_samples


def _write_img(path: Path, value: int) -> None:
    arr = np.full((16, 16), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_pair_samples(tmp_path: Path) -> None:
    sw = tmp_path / "SWIR"
    lw = tmp_path / "LWIR"
    lb = tmp_path / "labels"
    sw.mkdir()
    lw.mkdir()
    lb.mkdir()

    _write_img(sw / "0001.png", 50)
    _write_img(lw / "0001.png", 60)
    (lb / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    pairs = pair_samples(sw, lw, lb)
    assert len(pairs) == 1
    assert pairs[0].label is not None


def test_dataset_getitem(tmp_path: Path) -> None:
    sw = tmp_path / "SWIR"
    lw = tmp_path / "LWIR"
    lb = tmp_path / "labels"
    sw.mkdir()
    lw.mkdir()
    lb.mkdir()

    _write_img(sw / "0001.png", 50)
    _write_img(lw / "0001.png", 60)
    (lb / "0001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    ds = NSLSRDataset(sw, lw, lb)
    item = ds[0]
    assert item["swir"].shape == (1, 16, 16)
    assert item["lwir"].shape == (1, 16, 16)
    assert item["labels"].shape == (1, 5)
