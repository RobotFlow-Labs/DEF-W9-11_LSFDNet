from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SobelXY(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
        self.register_buffer("wx", kx)
        self.register_buffer("wy", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.wx, padding=1)
        gy = F.conv2d(x, self.wy, padding=1)
        return torch.abs(gx) + torch.abs(gy)


class OEFusionLoss(nn.Module):
    """Object Enhancement fusion loss aligned with LSFDNet paper equations (5)-(14)."""

    def __init__(self) -> None:
        super().__init__()
        self.sobel = SobelXY()

    @staticmethod
    def _mask_from_boxes(
        boxes: torch.Tensor, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        h, w = shape
        if boxes.numel() == 0:
            return torch.zeros((1, 1, h, w), device=device)
        mask = torch.zeros((1, 1, h, w), device=device)
        for box in boxes:
            # expected normalized xywh
            x, y, bw, bh = box[-4:].tolist()
            x1 = max(0, int((x - bw / 2.0) * w))
            y1 = max(0, int((y - bh / 2.0) * h))
            x2 = min(w, int((x + bw / 2.0) * w))
            y2 = min(h, int((y + bh / 2.0) * h))
            if x2 > x1 and y2 > y1:
                mask[:, :, y1:y2, x1:x2] = 1.0
        return mask

    def forward(
        self,
        swir: torch.Tensor,
        lwir: torch.Tensor,
        fused: torch.Tensor,
        labels: torch.Tensor | None,
        alpha: float = 0.5,
        beta: float = 0.5,
        sigma: float = 0.2,
        gamma: float = 2.7,
    ) -> torch.Tensor:
        # swir/lwir/fused: [B,1,H,W]
        lwir_th = torch.clamp(torch.pow(lwir, gamma), 0.0, 1.0)
        x_mean = 0.5 * swir + 0.5 * lwir_th
        x_mean_obj = 0.3 * swir + 0.7 * lwir_th

        g_swir = self.sobel(swir)
        g_mean = self.sobel(x_mean)
        g_fused = self.sobel(fused)
        g_obj = self.sobel(x_mean_obj)

        loss_int_global = F.l1_loss(fused, torch.max(swir, x_mean))
        loss_grad_global = F.l1_loss(g_fused, torch.max(g_swir, g_mean))
        loss_global = alpha * loss_int_global + (1.0 - alpha) * loss_grad_global

        if labels is None:
            labels = torch.zeros((0, 5), device=fused.device)
        mask = self._mask_from_boxes(
            labels.to(fused.device), (fused.shape[-2], fused.shape[-1]), fused.device
        )
        n_pixels = mask.sum()
        if n_pixels > 0:
            li = (
                F.l1_loss(fused, torch.max(swir, x_mean_obj), reduction="none") * mask
            ).sum() / n_pixels
            lg = (
                F.l1_loss(g_fused, torch.max(g_swir, g_obj), reduction="none") * mask
            ).sum() / n_pixels
            loss_obj = beta * li + (1.0 - beta) * lg
        else:
            loss_obj = fused.new_tensor(0.0)

        return (1.0 - sigma) * loss_global + sigma * loss_obj
