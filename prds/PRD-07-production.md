# PRD-07: Production Hardening

> Module: LSFDNet | Priority: P2
> Depends on: PRD-04
> Status: ⬜ Not started

## Objective
Harden LSFDNet for long-running production use with robust error handling, export paths, observability, and release artifacts.

## Context (from paper)
The model is intended for difficult maritime environments; production use requires resilience against noisy inputs, missing modality frames, and compute variability.

**Paper reference**: Section 5 (Conclusion and practical applicability).

## Acceptance Criteria
- [ ] Structured error taxonomy for invalid inputs, shape mismatch, and checkpoint errors.
- [ ] Export scripts for `pth -> safetensors -> ONNX` and optional TensorRT plan notes.
- [ ] Logging + metrics hooks for throughput, latency, and failure count.
- [ ] Release report includes metric deltas vs paper baselines.
- [ ] CI smoke pipeline validates inference + evaluation on sample data.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/export.py` | Export and artifact conversion commands | deployment | ~180 |
| `src/anima_lsfdnet/monitoring.py` | Runtime counters and latency probes | deployment | ~120 |
| `TRAINING_REPORT.md` | Repro + results summary | §4 | ~140 |

## Architecture Detail (from paper)

### Inputs
```text
checkpoint: Path
sample_input: Tensor[B,1,H,W] x 2 modalities
```

### Outputs
```text
artifacts/model.safetensors
artifacts/model.onnx
reports/training_report.md
```

### Algorithm
```python
def export_all(model, out_dir):
    save_pytorch(model, out_dir / "model.pth")
    save_safetensors(model, out_dir / "model.safetensors")
    export_onnx(model, out_dir / "model.onnx")
```

## Dependencies
```toml
onnx = ">=1.16"
safetensors = ">=0.4"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Representative eval subset | 50-100 pairs | `/mnt/forge-data/datasets/nslsr` | Baidu link |

## Test Plan
```bash
uv run python -m anima_lsfdnet.export --checkpoint ./PTH/net_LSFDNet.pth --out ./artifacts
uv run pytest tests -q
```

## References
- Paper: Section 4, Section 5
- Depends on: PRD-04
- Feeds into: release process
