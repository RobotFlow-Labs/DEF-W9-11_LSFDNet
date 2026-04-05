"""LSFDNet AnimaNode — SWIR/LWIR fusion serving node."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .model import create_model


class LSFDNetNode:
    """Minimal AnimaNode-compatible serving class for LSFDNet fusion.

    Provides setup_inference, process, and get_status methods.
    When anima_serve is available, subclass AnimaNode instead.
    """

    def __init__(self, weight_dir: str = "/data/weights", device: str = "auto") -> None:
        self.weight_dir = Path(weight_dir)
        self.device = self._resolve_device(device)
        self.model: torch.nn.Module | None = None
        self._ready = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def setup_inference(self) -> None:
        """Load model weights and prepare for inference."""
        self.model = create_model(device=self.device)
        self.model.eval()

        # Try loading weights from multiple formats in priority order
        weight_candidates = [
            self.weight_dir / "model.safetensors",
            self.weight_dir / "best.pth",
            self.weight_dir / "model.pth",
            self.weight_dir / "net_MMFusion.pth",
        ]
        for wpath in weight_candidates:
            if wpath.exists():
                state = torch.load(wpath, map_location=self.device, weights_only=True)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                elif isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
                self._ready = True
                return

        # No weights found — model runs with random init (useful for testing)
        self._ready = True

    def process(self, swir: np.ndarray, lwir: np.ndarray) -> dict:
        """Run inference on a SWIR/LWIR pair.

        Args:
            swir: Grayscale SWIR image, float32 [H, W] in [0, 1].
            lwir: Grayscale LWIR image, float32 [H, W] in [0, 1].

        Returns:
            dict with keys: fused (np.ndarray), runtime_ms (float), shape (tuple).
        """
        if self.model is None:
            raise RuntimeError("Call setup_inference() before process()")

        sw_t = torch.from_numpy(swir).float().unsqueeze(0).unsqueeze(0).to(self.device)
        lw_t = torch.from_numpy(lwir).float().unsqueeze(0).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            fused_t, _ = self.model(sw_t, lw_t)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        fused = fused_t.squeeze().cpu().numpy()
        return {
            "fused": fused,
            "runtime_ms": runtime_ms,
            "shape": fused.shape,
        }

    def get_status(self) -> dict:
        """Module-specific status fields."""
        return {
            "model_loaded": self.model is not None,
            "ready": self._ready,
            "device": self.device,
        }
