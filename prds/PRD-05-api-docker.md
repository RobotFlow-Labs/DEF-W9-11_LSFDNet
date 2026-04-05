# PRD-05: API and Docker Serving

> Module: LSFDNet | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose LSFDNet inference through a FastAPI service and a reproducible Docker serving stack.

## Context (from paper)
The model is practical for maritime perception; deployment requires service endpoints and health checks for edge/cloud integration.

**Paper reference**: Section 5 (Conclusion) practical deployment implications.

## Acceptance Criteria
- [ ] FastAPI exposes `/health`, `/ready`, `/predict`.
- [ ] `/predict` accepts paired SWIR/LWIR images and returns fused image + metadata.
- [ ] Docker image builds and runs with CUDA and CPU profiles.
- [ ] Compose stack includes health probes and volume mounts.
- [ ] Test: API integration test passes with synthetic inputs.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/api.py` | FastAPI app and request handlers | deployment | ~220 |
| `Dockerfile.serve` | Containerized inference runtime | deployment | ~80 |
| `docker-compose.serve.yml` | Service orchestration + health checks | deployment | ~80 |
| `.env.serve.example` | Runtime environment template | deployment | ~40 |

## Architecture Detail (from paper)

### Inputs
```text
multipart/form-data:
  swir: image file
  lwir: image file
```

### Outputs
```json
{
  "fused_image_base64": "...",
  "shape": [H, W],
  "runtime_ms": 0.0
}
```

### Algorithm
```python
@app.post("/predict")
def predict(swir_file, lwir_file):
    swir, lwir = decode_uploads(swir_file, lwir_file)
    fused, _ = model(swir, lwir)
    return serialize_response(fused)
```

## Dependencies
```toml
fastapi = ">=0.111"
uvicorn = ">=0.30"
python-multipart = ">=0.0.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Optional pretrained checkpoint | N/A | mounted at `/models/lsfdnet.pth` | Baidu link |

## Test Plan
```bash
uv run pytest tests/test_api.py -v
uv run uvicorn anima_lsfdnet.api:app --reload --port 8080
```

## References
- Paper: Section 5
- Depends on: PRD-03
- Feeds into: PRD-06, PRD-07
