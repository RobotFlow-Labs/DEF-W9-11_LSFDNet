# ANIMA LSFDNet Module (Wave-9)

This workspace contains:
- Paper source (`papers/2507.20574.pdf`)
- Reference implementation (`repositories/LSFDNet`)
- ANIMA planning artifacts (`ASSETS.md`, `prds/`, `tasks/`)
- Essential runnable module code (`src/anima_lsfdnet`)

## Quickstart

```bash
python -m pip install -e .[dev]
pytest
```

## Inference

```bash
python -m anima_lsfdnet.infer \
  --swir-dir ./data/NSLSR_test/SWIR \
  --lwir-dir ./data/NSLSR_test/LWIR \
  --out ./outputs
```

## Evaluation

```bash
python -m anima_lsfdnet.eval \
  --swir-dir ./data/NSLSR_test/SWIR \
  --lwir-dir ./data/NSLSR_test/LWIR \
  --fused-dir ./outputs/fused_images \
  --out ./reports
```
