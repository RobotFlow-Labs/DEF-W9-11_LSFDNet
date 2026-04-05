from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .io import to_base64_png
from .model import create_model


def create_app() -> Any:
    try:
        from fastapi import FastAPI, File, UploadFile
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Add fastapi + uvicorn to environment.") from exc

    app = FastAPI(title="LSFDNet API", version="0.1.0")
    model = create_model(device="cpu")
    model.eval()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/predict")
    async def predict(swir: UploadFile = File(...), lwir: UploadFile = File(...)) -> dict[str, Any]:
        from PIL import Image
        from io import BytesIO

        sw_arr = np.asarray(Image.open(BytesIO(await swir.read())).convert("L"), dtype=np.float32) / 255.0
        lw_arr = np.asarray(Image.open(BytesIO(await lwir.read())).convert("L"), dtype=np.float32) / 255.0
        sw = torch.from_numpy(sw_arr).unsqueeze(0).unsqueeze(0)
        lw = torch.from_numpy(lw_arr).unsqueeze(0).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            fused, _ = model(sw, lw)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        fused_np = fused.squeeze().cpu().numpy()
        return {
            "fused_image_base64": to_base64_png(fused_np),
            "shape": list(fused_np.shape),
            "runtime_ms": runtime_ms,
        }

    return app


try:  # pragma: no cover - runtime dependency gate
    app = create_app()
except RuntimeError:
    app = None
