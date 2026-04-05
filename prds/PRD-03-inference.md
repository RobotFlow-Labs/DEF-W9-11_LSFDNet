# PRD-03: Inference Pipeline

> Module: LSFDNet | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Deliver a CLI inference pipeline that loads checkpoints, runs SWIR/LWIR fusion, and writes fused outputs and metadata.

## Context (from paper)
LSFDNet generates fused images and jointly supports downstream detection. Operational inference must support test-time fusion at NSLSR resolution and optional size overrides.

**Paper reference**: Section 3.1 and Section 4.1.
- End-to-end fusion/detection architecture.
- Evaluated on NSLSR split and fusion metrics on selected test subset.

## Acceptance Criteria
- [ ] `python -m anima_lsfdnet.infer` supports folder-based SWIR/LWIR inference.
- [ ] Supports checkpoint loading and CPU/GPU selection.
- [ ] Saves fused image and per-sample JSON metadata.
- [ ] Handles missing paired frames robustly.
- [ ] Test: `uv run pytest tests/test_infer.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/infer.py` | Inference CLI and batch runner | §3.1 | ~220 |
| `src/anima_lsfdnet/io.py` | Image pair discovery and save utilities | §4.1 | ~140 |
| `tests/test_infer.py` | Inference smoke test | — | ~70 |

## Architecture Detail (from paper)

### Inputs
```text
swir_dir: Path
lwir_dir: Path
checkpoint: Optional[Path]
img_size: Tuple[int, int]
```

### Outputs
```text
fused_images/*.png
predictions.jsonl
```

### Algorithm
```python
# Reference: repositories/LSFDNet/LSFDNet/test.py

for pair in paired_image_iterator(swir_dir, lwir_dir):
    swir, lwir = preprocess(pair)
    fused, _ = model(swir, lwir)
    save_fused(fused)
    write_metadata(pair_id, shape, runtime_ms)
```

## Dependencies
```toml
numpy = ">=1.25"
Pillow = ">=10.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| NSLSR test SWIR/LWIR folders | 118 or 361 pairs | `/mnt/forge-data/datasets/nslsr/NSLSR_test` | Baidu link |

## Test Plan
```bash
uv run pytest tests/test_infer.py -v
uv run python -m anima_lsfdnet.infer --swir-dir ./sample/SWIR --lwir-dir ./sample/LWIR --out ./outputs/fused
```

## References
- Paper: Section 3.1, Section 4.1
- Reference impl: `repositories/LSFDNet/LSFDNet/test.py`
- Depends on: PRD-02
- Feeds into: PRD-04, PRD-05
