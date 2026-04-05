# PRD-04: Evaluation and Benchmarks

> Module: LSFDNet | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement paper-aligned fusion and detection evaluation scripts that compare local runs to published LSFDNet baselines.

## Context (from paper)
Paper benchmarks include fusion metrics (EN, SF, SD, SCD, VIF, Qabf) and detection metrics (mAP50, mAP50:95) on NSLSR.

**Paper reference**: Section 4.1, Table 1, Table 2, Section 4.4.

## Acceptance Criteria
- [ ] Compute fusion metrics on fused outputs and report summary table.
- [ ] Compute detection metrics from prediction/label files.
- [ ] Emit comparison report against paper values in `ASSETS.md`.
- [ ] Include ablation report template for OE loss and MLCF.
- [ ] Test: `uv run pytest tests/test_metrics.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_lsfdnet/metrics.py` | Fusion + detection metric functions | §4.1, Table 1/2 | ~220 |
| `src/anima_lsfdnet/eval.py` | Evaluation runner + report writer | §4.2-4.4 | ~180 |
| `tests/test_metrics.py` | Metric sanity tests | — | ~90 |

## Architecture Detail (from paper)

### Inputs
```text
gt_pairs: SWIR/LWIR (+labels)
fused_outputs: fused images
pred_labels: optional detector outputs
```

### Outputs
```text
reports/eval_report.md
reports/eval_metrics.json
```

### Algorithm
```python
# Paper Section 4
fusion_scores = compute_fusion_metrics(fused, swir, lwir)
det_scores = compute_detection_metrics(pred_boxes, gt_boxes)
report = compare_to_paper(fusion_scores, det_scores, baselines)
```

## Dependencies
```toml
numpy = ">=1.25"
scipy = ">=1.11"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| NSLSR fusion eval subset | 118 pairs | `/mnt/forge-data/datasets/nslsr/NSLSR_test` | Baidu link |
| NSLSR detection eval set | 361 pairs + labels | `/mnt/forge-data/datasets/nslsr/NSLSR_val`/`test` | Baidu link |

## Test Plan
```bash
uv run pytest tests/test_metrics.py -v
uv run python -m anima_lsfdnet.eval --config configs/paper.toml --pred-dir ./outputs
```

## References
- Paper: Section 4.1-4.4, Table 1-4
- Reference impl: `repositories/LSFDNet/LSFDNet/core/Metric_fusion`
- Depends on: PRD-03
- Feeds into: PRD-07
