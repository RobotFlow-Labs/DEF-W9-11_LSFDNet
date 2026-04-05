import torch

from anima_lsfdnet.model import LSFDNetFusionCore


def test_model_forward_shape() -> None:
    model = LSFDNetFusionCore()
    sw = torch.randn(2, 1, 128, 128)
    lw = torch.randn(2, 1, 128, 128)
    fused, feat = model(sw, lw)
    assert fused.shape == (2, 1, 128, 128)
    assert feat.shape == (2, 8, 128, 128)
