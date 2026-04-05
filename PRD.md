# LSFDNet -- Master Build Plan

> Module: LSFDNet (ANIMA Wave-9)
> Paper: arXiv 2507.20574 -- ACM Multimedia 2025
> Type: perception (SWIR/LWIR image fusion + ship detection)

## Overview

LSFDNet is a single-stage fusion and detection network for maritime ship detection
using Short-Wave Infrared (SWIR) and Long-Wave Infrared (LWIR) imagery. The ANIMA
module implements the fusion core (MTFE + MLCF + Decoder) with the Object Enhancement
loss, inference pipeline, evaluation metrics, API serving, and ROS2 integration.

The detection branch uses a YOLOv12-based backbone (not reimplemented here -- deferred
to training phase with real data and the reference repository's MMYOLO integration).

## Build Plan

| PRD | Title | Priority | Status | File |
|-----|-------|----------|--------|------|
| PRD-01 | Foundation and Config | P0 | DONE | [prds/PRD-01-foundation.md](prds/PRD-01-foundation.md) |
| PRD-02 | Core Model Architecture | P0 | DONE | [prds/PRD-02-core-model.md](prds/PRD-02-core-model.md) |
| PRD-03 | Inference Pipeline | P0 | DONE | [prds/PRD-03-inference.md](prds/PRD-03-inference.md) |
| PRD-04 | Evaluation and Benchmarks | P1 | DONE | [prds/PRD-04-evaluation.md](prds/PRD-04-evaluation.md) |
| PRD-05 | API and Docker Serving | P1 | DONE | [prds/PRD-05-api-docker.md](prds/PRD-05-api-docker.md) |
| PRD-06 | ROS2 Integration | P1 | SCAFFOLD | [prds/PRD-06-ros2-integration.md](prds/PRD-06-ros2-integration.md) |
| PRD-07 | Production Hardening | P2 | DONE | [prds/PRD-07-production.md](prds/PRD-07-production.md) |

### Status Key

- **DONE**: Source files, configs, and tests exist in the scaffold.
- **SCAFFOLD**: Placeholder files exist but need real implementation (requires ROS2 env).
- **BLOCKED**: Waiting on external dependency (dataset, weights, hardware).

## Architecture Summary

```
SWIR image -----> Base Feature Extractor ----> Fusion Feature Extractor ----> F_tilde_SW
                  (3 conv: 1->4->8->8)         (7 conv: 8->16->...->8)          |
LWIR image -----> Base Feature Extractor ----> Fusion Feature Extractor ----> F_tilde_LW
                  (shared weights)              (separate weights)               |
                                                                                 v
                                                                    MLCF (3 MFA blocks)
                                                                    - MFA_high (full res)
                                                                    - MFA_low (half res)
                                                                    - MFA_cross (combine)
                                                                                 |
                                                                                 v
                                                              F_f (8-ch fused feature)
                                                              |                  |
                                                              v                  v
                                                    Fusion Decoder       Detection Branch
                                                    (8->16->32->16->4->1)  (YOLOv12, deferred)
                                                              |
                                                              v
                                                    Fused image I_f [1,H,W]
```

## Key Design Decisions

1. **Fusion-only core**: The scaffold implements the full fusion pathway (MTFE + MLCF +
   Decoder + OE loss). The detection branch (YOLOv12-based) is deferred to training
   phase because it requires the MMYOLO framework and real NSLSR data.

2. **Cross-task coupling placeholder**: The model.py `forward()` accepts an optional
   `det_attention` tensor for detection-to-fusion feedback (0.6/0.4 weighted sigmoid
   gating). This is ready for integration once the detection branch is built.

3. **MFA implementation**: Uses patch-based embedding -> self-attention -> cross-attention
   -> MLP -> convolutional decode, matching the paper's Eq. (1)-(4).

4. **OE loss**: Full implementation of Eq. (5)-(14) with Sobel gradients, gamma correction,
   global+object loss components, and bounding-box masking.

## Datasets

| Dataset | Pairs | Resolution | Split | Server Path |
|---------|-------|-----------|-------|-------------|
| NSLSR | 1,205 | 640x512 | 844 train / 361 test | /mnt/forge-data/datasets/nslsr |
| ISD | 1,045 | 300x300 | 940 train / 105 test | /mnt/forge-data/datasets/isd |

Both are MISSING and require Baidu Pan download (manual, scp from Mac).

## Weights

| Checkpoint | Server Path |
|------------|-------------|
| net_LSFDNet.pth (joint) | /mnt/forge-data/models/lsfdnet/net_LSFDNet.pth |
| net_MMFusion.pth (fusion) | /mnt/forge-data/models/lsfdnet/net_MMFusion.pth |

Both are MISSING and require Baidu Pan download.

## Training Configuration

Training outputs go to `/mnt/artifacts-datai/` per ANIMA rules:
- Checkpoints: `/mnt/artifacts-datai/checkpoints/lsfdnet/`
- Logs: `/mnt/artifacts-datai/logs/lsfdnet/`
- TensorBoard: `/mnt/artifacts-datai/tensorboard/lsfdnet/`
- Exports: `/mnt/artifacts-datai/exports/lsfdnet/`

## Verification

```bash
cd /mnt/forge-data/modules/05_wave9/11_LSFDNet
source .venv/bin/activate
uv run pytest tests/ -v
```
