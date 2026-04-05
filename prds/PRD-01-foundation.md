# PRD-01: Foundation and Config

> Module: LSFDNet | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Establish a reproducible ANIMA-ready project skeleton (config, data contracts, loaders, package layout, and smoke tests) for LSFDNet.

## Context (from paper)
LSFDNet is trained end-to-end on registered SWIR/LWIR pairs with task-coupled fusion and detection. Reproducible input formatting and aligned SWIR/LWIR loading are mandatory.

**Paper reference**: Section 3.1 and Section 4.1.
- “Given a pair of registered SWIR and LWIR images as input…”
- “We construct a training set with 844 images and a testing set with 361 images from the NSLSR dataset.”

## Acceptance Criteria
- [ ] Config system loads defaults and paper profile (`configs/default.toml`, `configs/paper.toml`).
- [ ] Dataset loader yields aligned SWIR/LWIR tensors and optional labels.
- [ ] Reproducibility controls are exposed (seed, deterministic, warmup, total iterations).
- [ ] Unit tests for config loading and synthetic dataset loading pass.
- [ ] Test: `uv run pytest tests/test_config.py tests/test_dataset.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/config.py` | Typed runtime config + TOML loader | §4.1 | ~160 |
| `src/anima_lsfdnet/dataset.py` | NSLSR-compatible SWIR/LWIR dataset adapter | §3.1, §4.1 | ~200 |
| `configs/default.toml` | Default local settings | §4.1 | ~80 |
| `configs/paper.toml` | Paper-faithful hyperparameter profile | §4.1 | ~60 |
| `tests/test_config.py` | Config loading tests | — | ~50 |
| `tests/test_dataset.py` | Dataset synthetic I/O tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
```text
swir: Tensor[B,1,H,W]
lwir: Tensor[B,1,H,W]
labels: Optional[Tensor[N,5]]  # cls + normalized bbox
```

### Outputs
```text
batch: dict[str, Tensor]
```

### Algorithm
```python
# Paper Sections 3.1, 4.1
# Reference: repositories/LSFDNet/LSFDNet/data/FusionDet_dataset.py

class NSLSRDataset(Dataset):
    """Aligned SWIR/LWIR pair loader with optional detection labels."""
    def __getitem__(self, idx):
        swir = load_gray(self.swir_paths[idx])
        lwir = load_gray(self.lwir_paths[idx])
        target = load_optional_yolo_label(self.label_paths[idx])
        return {"swir": swir, "lwir": lwir, "target": target}
```

## Dependencies
```toml
torch = ">=2.1"
torchvision = ">=0.16"
tomli = ">=2.0"  # py<3.11 compatibility
pillow = ">=10.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| NSLSR train split | 844 pairs | `/mnt/forge-data/datasets/nslsr/NSLSR_train` | Baidu link in `ASSETS.md` |
| NSLSR val/test split | 361 pairs total | `/mnt/forge-data/datasets/nslsr/NSLSR_val`, `/mnt/forge-data/datasets/nslsr/NSLSR_test` | Baidu link in `ASSETS.md` |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_dataset.py -v
uv run python -m anima_lsfdnet.checks --config configs/default.toml
```

## References
- Paper: Section 3.1 “Overview”, Section 4.1 “Dataset and Implementation Details”
- Reference impl: `repositories/LSFDNet/LSFDNet/data/FusionDet_dataset.py`
- Depends on: None
- Feeds into: PRD-02
