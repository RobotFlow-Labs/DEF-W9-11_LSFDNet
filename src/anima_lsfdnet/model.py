from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock, MFAttentionBlock


class FeatureExtractorBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 4),
            ConvBlock(4, 8),
            ConvBlock(8, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureExtractorMulNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(8, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MulLayerFusion(nn.Module):
    """MLCF approximation: high-level fusion, low-level fusion, and cross-scale fusion."""

    def __init__(self, channels: int = 8, patch_size: int = 8, heads: int = 8) -> None:
        super().__init__()
        self.down_s = nn.Sequential(
            ConvBlock(channels, channels, stride=2, prelu=True),
            ConvBlock(channels, channels, prelu=True),
            ConvBlock(channels, channels, prelu=True),
        )
        self.down_l = nn.Sequential(
            ConvBlock(channels, channels, stride=2, prelu=True),
            ConvBlock(channels, channels, prelu=True),
            ConvBlock(channels, channels, prelu=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(channels, channels, prelu=True),
            ConvBlock(channels, channels, prelu=True),
        )

        self.mfa_high = MFAttentionBlock(channels=channels, patch_size=patch_size, heads=heads)
        self.mfa_low = MFAttentionBlock(channels=channels, patch_size=patch_size, heads=heads)
        self.mfa_cross = MFAttentionBlock(channels=channels, patch_size=patch_size, heads=heads)

    def forward(self, feat_s: torch.Tensor, feat_l: torch.Tensor) -> torch.Tensor:
        fused_h = self.mfa_high(feat_s, feat_l)
        low_s = self.down_s(feat_s)
        low_l = self.down_l(feat_l)
        fused_l = self.up(self.mfa_low(low_s, low_l))
        return self.mfa_cross(fused_h, fused_l)


class FusionDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(8, 8, prelu=True),
            ConvBlock(8, 16, prelu=True),
            ConvBlock(16, 16, prelu=True),
            ConvBlock(16, 32, prelu=True),
            ConvBlock(32, 32, prelu=True),
            ConvBlock(32, 16, prelu=True),
            ConvBlock(16, 16, prelu=True),
            ConvBlock(16, 4, prelu=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSFDNetFusionCore(nn.Module):
    """
    Essential LSFDNet fusion pathway implementation.

    Returns:
    - fused image tensor [B,1,H,W]
    - intermediate fused feature F_f [B,8,H,W]
    """

    def __init__(self, patch_size: int = 8, heads: int = 8):
        super().__init__()
        self.base = FeatureExtractorBase()
        self.fsw = FeatureExtractorMulNet()
        self.flw = FeatureExtractorMulNet()
        self.mlc_fusion = MulLayerFusion(channels=8, patch_size=patch_size, heads=heads)
        self.decoder = FusionDecoder()

    def forward(self, swir: torch.Tensor, lwir: torch.Tensor, det_attention: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sw_base = self.base(swir)
        lw_base = self.base(lwir)
        fsw = self.fsw(sw_base)
        flw = self.flw(lw_base)
        fused_feature = self.mlc_fusion(fsw, flw)

        if det_attention is not None:
            fused_feature = 0.6 * fused_feature + 0.4 * fused_feature * torch.sigmoid(det_attention)

        fused_image = self.decoder(fused_feature)
        return fused_image, fused_feature


def create_model(device: str = "cpu") -> LSFDNetFusionCore:
    model = LSFDNetFusionCore()
    return model.to(device)
