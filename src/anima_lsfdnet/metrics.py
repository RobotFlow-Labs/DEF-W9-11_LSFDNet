from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-8


def entropy(image: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(np.clip(image, 0.0, 1.0), bins=bins, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob /= max(prob.sum(), 1.0)
    prob = prob[prob > 0]
    value = float(-np.sum(prob * np.log2(prob + EPS)))
    return max(0.0, value)


def spatial_frequency(image: np.ndarray) -> float:
    rf = np.diff(image, axis=0)
    cf = np.diff(image, axis=1)
    return float(math.sqrt(np.mean(rf**2) + np.mean(cf**2)))


def standard_deviation(image: np.ndarray) -> float:
    return float(np.std(image))


def scd(fused: np.ndarray, swir: np.ndarray, lwir: np.ndarray) -> float:
    d1 = fused - swir
    d2 = fused - lwir
    num = np.sum((d1 - d1.mean()) * (d2 - d2.mean()))
    den = math.sqrt(np.sum((d1 - d1.mean()) ** 2) * np.sum((d2 - d2.mean()) ** 2) + EPS)
    return float(num / den)


def qabf(fused: np.ndarray, swir: np.ndarray, lwir: np.ndarray) -> float:
    # Lightweight proxy: edge retention ratio against stronger edge map.
    def grad(x: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:, 1:] = np.abs(x[:, 1:] - x[:, :-1])
        gy[1:, :] = np.abs(x[1:, :] - x[:-1, :])
        return gx + gy

    gf = grad(fused)
    gref = np.maximum(grad(swir), grad(lwir))
    return float(np.mean(np.minimum(gf, gref) / (np.maximum(gref, EPS))))


@dataclass
class FusionMetrics:
    en: float
    sf: float
    sd: float
    scd: float
    qabf: float


def compute_fusion_metrics(fused: np.ndarray, swir: np.ndarray, lwir: np.ndarray) -> FusionMetrics:
    return FusionMetrics(
        en=entropy(fused),
        sf=spatial_frequency(fused),
        sd=standard_deviation(fused),
        scd=scd(fused, swir, lwir),
        qabf=qabf(fused, swir, lwir),
    )


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + EPS))
