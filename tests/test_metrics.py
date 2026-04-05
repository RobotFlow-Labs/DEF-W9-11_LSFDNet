import numpy as np

from anima_lsfdnet.metrics import compute_fusion_metrics, entropy, spatial_frequency


def test_entropy_non_negative() -> None:
    img = np.zeros((16, 16), dtype=np.float32)
    assert entropy(img) >= 0.0


def test_spatial_frequency_zero_for_constant() -> None:
    img = np.ones((16, 16), dtype=np.float32)
    assert spatial_frequency(img) == 0.0


def test_compute_fusion_metrics() -> None:
    sw = np.zeros((16, 16), dtype=np.float32)
    lw = np.ones((16, 16), dtype=np.float32) * 0.5
    fu = np.ones((16, 16), dtype=np.float32) * 0.25
    m = compute_fusion_metrics(fu, sw, lw)
    assert m.en >= 0.0
    assert m.sd >= 0.0
