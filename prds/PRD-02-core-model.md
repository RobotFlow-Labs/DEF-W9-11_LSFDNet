# PRD-02: Core Model Architecture

> Module: LSFDNet | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement the core LSFDNet fusion architecture (MTFE + MLCF + fusion decoder) with paper-consistent interfaces and a runnable forward pass.

## Context (from paper)
LSFDNet couples fusion and detection. The fusion side uses MTFE followed by MLCF with three MFA blocks to aggregate multimodal, multiscale, and multitask cues.

**Paper reference**: Section 3.2 and Section 3.3.
- “The MLCF module… is composed of three Multi-Feature Attention (MFA) blocks.”
- “The Decoder block… expand[s] the features from 8 channels to 16 channels and then reduce[s] them back to 8 channels.”

## Acceptance Criteria
- [ ] Model forward consumes SWIR/LWIR tensors and returns fused image.
- [ ] MLCF contains three MFA stages (`high`, `low`, `cross-scale`).
- [ ] Intermediate fused feature (F_f) is available for detection branch coupling.
- [ ] OE loss-compatible output range and shape are correct.
- [ ] Test: `uv run pytest tests/test_model.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/model.py` | MTFE + MLCF + decoder implementation | §3.2, §3.3 | ~320 |
| `src/anima_lsfdnet/blocks.py` | Reusable Conv/MFA modules | Eq. (1)-(4) | ~180 |
| `tests/test_model.py` | Forward shape + determinism smoke tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
```text
swir: Tensor[B,1,H,W]
lwir: Tensor[B,1,H,W]
```

### Outputs
```text
fused_image: Tensor[B,1,H,W]
fused_feature: Tensor[B,8,H,W]
```

### Algorithm
```python
# Paper Sections 3.2-3.3, Eq. (1)-(4)
# Reference: repositories/LSFDNet/LSFDNet/archs/MMFusion_arch.py

class LSFDNetFusionCore(nn.Module):
    def forward(self, swir, lwir):
        fsw_base, flw_base = base_extract(swir), base_extract(lwir)
        fsw, flw = fusion_extract(fsw_base), fusion_extract(flw_base)
        f_high = mfa_high(fsw, flw)
        f_low = upsample(mfa_low(down(fsw), down(flw)))
        f_fused = mfa_cross_scale(f_high, f_low)
        fused_image = decoder(f_fused)
        return fused_image, f_fused
```

## Dependencies
```toml
torch = ">=2.1"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Synthetic input for unit tests | N/A | generated in test runtime | N/A |
| NSLSR real samples | 640x512 grayscale pairs | `/mnt/forge-data/datasets/nslsr/...` | Baidu link in `ASSETS.md` |

## Test Plan
```bash
uv run pytest tests/test_model.py -v
uv run python -c "import torch; from anima_lsfdnet.model import LSFDNetFusionCore; m=LSFDNetFusionCore(); y,_=m(torch.randn(2,1,320,320), torch.randn(2,1,320,320)); print(y.shape)"
```

## References
- Paper: Section 3.2 “Multi-Task Feature Extraction”, Section 3.3 “Multi-Level Cross-Fusion Module”, Eq. (1)-(4)
- Reference impl: `repositories/LSFDNet/LSFDNet/archs/MMFusion_arch.py`, `MMYOLO_arch.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04
