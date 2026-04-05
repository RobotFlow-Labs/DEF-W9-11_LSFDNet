# LSFDNet -- ANIMA Module Reference

## Paper Summary

**Title**: LSFDNet: A Single-Stage Fusion and Detection Network for Ships Using SWIR and LWIR
**Authors**: Yanyin Guo, Runxuan An, Junwei Li, Zhiyuan Zhang
**Venue**: ACM Multimedia 2025 (MM '25, Dublin, Ireland)
**ArXiv**: 2507.20574
**Code**: https://github.com/Yanyin-Guo/LSFDNet

LSFDNet is the first single-stage network that jointly performs SWIR-LWIR image fusion
and ship object detection in an end-to-end framework. It targets maritime ship detection
under challenging conditions (fog, low-light, sea surface noise) where single-modality
approaches fail. The key insight is that SWIR images preserve texture/detail while LWIR
images provide lighting-invariant thermal contrast; fusing them yields imagery that is
simultaneously high-contrast and detail-rich.

### Architecture (Section 3)

The network has three main components:

1. **Multi-Task Feature Extraction (MTFE)** (Section 3.2):
   - Base Feature Extractor: 3 conv layers (1->4->8->8 channels, kernel 3x3, stride 1)
     producing shared shallow features F_SW and F_LW from each modality.
   - Fusion Feature Extractor: 7 conv layers (8->16->16->32->32->16->16->8) expanding
     and compressing features to produce F_tilde_SW and F_tilde_LW.
   - Detection Feature Extractor: YOLOv12-based backbone with four sub-modules --
     Shallow Feature Extraction, Fusion Feature Augmentation, Deep Feature Extraction,
     and Multimodal Feature Aggregation. Uses A2C2f blocks for area attention.

2. **Multi-Level Cross-Fusion (MLCF)** (Section 3.3):
   - Three Multi-Feature Attention (MFA) blocks processing features at multiple levels:
     - MFA_high: aggregates F_tilde_SW and F_tilde_LW at original resolution -> F_f_H
     - MFA_low: aggregates downsampled features, then upsamples -> F_f_L
     - MFA_cross: combines F_f_H and F_f_L into preliminary fusion feature F_f
   - Each MFA block uses patch-based self-attention + cross-attention (Eq. 1-4):
     patches of size p x p are vectorized, projected to Q/K/V, processed through
     self-attention per modality, then cross-attention across modalities, followed
     by an MLP and a convolutional decoder block.
   - F_f is fed back into the detection branch; detection attention F_det_attn
     is fed back into the fusion branch (cross-task coupling).

3. **Fusion Decoder** (Section 3.3):
   - 4 conv layers: 8->16->16->32->32->16->16->4 channels, then final Conv2d(4,1,3).
   - Expands features, reduces back, outputs single-channel fused image I_f<-det.

### Loss Function (Section 3.4)

Total loss: L = (1-lambda) * L_f + lambda * L_det (Eq. 5)

- L_det: YOLOv12 detection loss
- L_f: Object Enhancement (OE) fusion loss (Eq. 7-14):
  - Global loss L_f^global = (1-alpha) * L_grad_global + alpha * L_intensity_global
    - L_grad_global: L1 of Sobel(fused) vs max(Sobel(SWIR), Sobel(mean_image))
    - L_intensity_global: L1 of fused vs max(SWIR, gamma-corrected LWIR mean)
  - Object loss L_f^object: same structure but masked to object bounding boxes
  - L_f = (1-sigma) * L_f^global + sigma * L_f^object
  - Gamma correction on LWIR: I'_LW = 255 * (I_LW / 255)^gamma (Eq. 6)

### NSLSR Dataset (Section 3.5)

- 1,205 registered SWIR/LWIR pairs, 640x512 resolution
- 2,818 annotated ship objects
- SWIR camera: InGaAs FPA, 0.9-1.7 um, uncooled
- LWIR camera: VOx FPA, 8-14 um, uncooled
- Train/test split: 844/361 (9:1 ratio)
- Fusion eval uses 118 test images; detection eval uses all 361 test images

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| optimizer | Adam | Paper Section 4.1 |
| learning_rate | 1e-4 | Paper Section 4.1 |
| lr_schedule | linear decay | Paper Section 4.1 |
| warmup_iterations | 500 | Paper Section 4.1 |
| total_iterations | 30,000 (detection), 300,000 (fusion-only) | Paper + repo configs |
| batch_size | 8 | Paper Section 4.1 |
| gamma (LWIR correction) | 2.7 | Paper Eq. 6 + repo config |
| sigma (OE loss balance) | 0.2 (paper), 0.8 (repo default) | Paper Section 4.1 |
| alpha (global grad/int balance) | 0.5 | Paper Section 4.1 |
| beta (object grad/int balance) | 0.5 | Paper Section 4.1 |
| input_resolution | 640x512 (paper), 320x320 (repo updated) | Paper + repo README |
| base_channels | 8 | Paper Section 3.2 |
| patch_size (MFA) | 8 (inferred from code) | Repo implementation |
| attention_heads | 8 (inferred from code) | Repo implementation |
| GPU | NVIDIA RTX 4090 | Paper Section 4.1 |

## Expected Metrics

### Fusion (NSLSR, 118 test images -- Table 1)

| Metric | Paper Value | Target |
|--------|-------------|--------|
| EN (entropy) | 7.181 | >= 7.10 |
| SF (spatial frequency) | 21.022 | >= 20.5 |
| SD (standard deviation) | 64.723 | >= 64.0 |
| SCD (sum of corr. diff.) | 1.427 | >= 1.40 |
| VIF (visual info fidelity) | 0.611 | >= 0.60 |
| Qabf (edge-based metric) | 0.520 | >= 0.50 |

### Detection (NSLSR, 361 test images -- Table 2)

| Metric | Paper Value | Target |
|--------|-------------|--------|
| mAP@0.5 | 0.962 | >= 0.96 |
| mAP@0.5:0.95 | 0.770 | >= 0.77 |
| Precision | 0.934 | >= 0.93 |
| Recall | 0.887 | >= 0.88 |

### Updated checkpoint (repo README, 320x320)

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.9855 |
| mAP@0.5:0.95 | 0.7883 |

## Dataset Requirements

| Dataset | Size | Split | Path | Status |
|---------|------|-------|------|--------|
| NSLSR | 1,205 SWIR/LWIR pairs | train=844 / test=361 | /mnt/forge-data/datasets/nslsr | MISSING |
| ISD | 1,045 pairs | train=940 / test=105 | /mnt/forge-data/datasets/isd | MISSING |

**Download**: Both datasets are on Baidu Pan (see ASSETS.md for links and codes).
Manual download required -- transfer via scp from Mac.

## Model / Weights Requirements

| Weight | Path | Status |
|--------|------|--------|
| LSFDNet (joint MMYOLO) | /mnt/forge-data/models/lsfdnet/net_LSFDNet.pth | MISSING |
| MMFusion branch | /mnt/forge-data/models/lsfdnet/net_MMFusion.pth | MISSING |

**Download**: Baidu Pan (see ASSETS.md for link and code: x7sf).
Manual download required -- transfer via scp from Mac.

## Build Commands

```bash
cd /mnt/forge-data/modules/05_wave9/11_LSFDNet
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv sync

# Run tests
uv run pytest tests/ -v

# Smoke inference (synthetic)
uv run python -m anima_lsfdnet.infer --swir-dir ./sample/SWIR --lwir-dir ./sample/LWIR --out ./outputs

# Training (requires NSLSR dataset)
CUDA_VISIBLE_DEVICES=0 uv run python -m anima_lsfdnet.train --config configs/paper.toml --epochs 100
```

## File Map

```
src/anima_lsfdnet/
  __init__.py        -- package exports
  config.py          -- TOML config loader + dataclasses
  dataset.py         -- NSLSR/ISD aligned SWIR/LWIR dataset
  model.py           -- MTFE + MLCF + Decoder (fusion core)
  blocks.py          -- ConvBlock, PatchEmbed, MFAttentionBlock
  losses.py          -- OE fusion loss (Eq. 5-14)
  train.py           -- training loop entry point
  infer.py           -- CLI inference pipeline
  eval.py            -- evaluation runner + report
  metrics.py         -- fusion + detection metric functions
  api.py             -- FastAPI serving endpoints
  io.py              -- image pair discovery utilities
  export.py          -- pth -> safetensors -> ONNX export
  monitoring.py      -- runtime counters and latency probes
  checks.py          -- asset and config validation

configs/
  default.toml       -- local dev defaults (320x320, batch 2, CPU)
  paper.toml         -- paper-faithful hyperparameters (640x512, batch 8, CUDA)
  debug.toml         -- quick smoke test config (2 steps)

tests/
  test_config.py     -- config loading tests
  test_dataset.py    -- dataset I/O tests
  test_model.py      -- forward pass shape tests
  test_infer.py      -- inference pipeline smoke test
  test_metrics.py    -- metric sanity tests
  test_api.py        -- API integration test
```
