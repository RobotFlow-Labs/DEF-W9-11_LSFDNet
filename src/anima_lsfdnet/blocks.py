from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        prelu: bool = False,
    ):
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.ReplicationPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU() if prelu else nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, out_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_chans, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, n_tokens, dim = x.shape
        h_tokens = max(1, height // self.patch_size)
        w_tokens = max(1, width // self.patch_size)
        x = x.transpose(1, 2).reshape(batch, dim, h_tokens, w_tokens)
        return self.proj(x)


class MFAttentionBlock(nn.Module):
    """
    Simplified MFA block inspired by LSFDNet Eq. (1)-(4):
    self-attention on each modality + cross-attention between modalities.
    """

    def __init__(self, channels: int = 8, patch_size: int = 8, heads: int = 8):
        super().__init__()
        embed_dim = max(64, channels * patch_size * patch_size)
        if embed_dim % heads != 0:
            embed_dim = heads * ((embed_dim // heads) + 1)

        self.swir_enc = nn.Sequential(
            ConvBlock(channels, channels // 2, prelu=True),
            ConvBlock(channels // 2, channels // 2, prelu=True),
        )
        self.lwir_enc = nn.Sequential(
            ConvBlock(channels, channels // 2, prelu=True),
            ConvBlock(channels // 2, channels // 2, prelu=True),
        )

        self.embed_s = PatchEmbed(channels // 2, embed_dim, patch_size)
        self.embed_l = PatchEmbed(channels // 2, embed_dim, patch_size)
        self.unembed_s = PatchUnEmbed(channels // 2, embed_dim, patch_size)
        self.unembed_l = PatchUnEmbed(channels // 2, embed_dim, patch_size)

        self.self_s = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.self_l = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.cross_s = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.cross_l = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

        self.norm1s = nn.LayerNorm(embed_dim)
        self.norm1l = nn.LayerNorm(embed_dim)
        self.norm2s = nn.LayerNorm(embed_dim)
        self.norm2l = nn.LayerNorm(embed_dim)

        self.ffn_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_s_cross = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_l_cross = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.decode = nn.Sequential(
            ConvBlock(channels, channels, prelu=True),
            ConvBlock(channels, channels * 2, prelu=True),
            ConvBlock(channels * 2, channels * 2, prelu=True),
            ConvBlock(channels * 2, channels, prelu=True),
        )

    def forward(self, swir: torch.Tensor, lwir: torch.Tensor) -> torch.Tensor:
        h, w = swir.shape[-2:]
        s = self.swir_enc(swir)
        lw = self.lwir_enc(lwir)

        s_tok = self.embed_s(s)
        l_tok = self.embed_l(lw)

        s_attn, _ = self.self_s(
            self.norm1s(s_tok), self.norm1s(s_tok), self.norm1s(s_tok), need_weights=False
        )
        l_attn, _ = self.self_l(
            self.norm1l(l_tok), self.norm1l(l_tok), self.norm1l(l_tok), need_weights=False
        )
        s_tok = s_tok + self.ffn_s(s_attn)
        l_tok = l_tok + self.ffn_l(l_attn)

        s_cross, _ = self.cross_s(self.norm2s(s_tok), l_tok, l_tok, need_weights=False)
        l_cross, _ = self.cross_l(self.norm2l(l_tok), s_tok, s_tok, need_weights=False)
        s_tok = s_tok + self.ffn_s_cross(s_cross)
        l_tok = l_tok + self.ffn_l_cross(l_cross)

        s_out = self.unembed_s(s_tok, h, w)
        l_out = self.unembed_l(l_tok, h, w)
        return self.decode(torch.cat([s_out, l_out], dim=1))
