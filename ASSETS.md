# LSFDNet — Asset Manifest

## Paper
- Title: LSFDNet: A Single-Stage Fusion and Detection Network for Ships Using SWIR and LWIR
- ArXiv: 2507.20574
- Authors: Yanyin Guo, Runxuan An, Junwei Li, Zhiyuan Zhang
- Conference: ACM Multimedia 2025

## Status: ALMOST

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|-------|------|--------|---------------|--------|
| LSFDNet (joint MMYOLO) | N/A (not published in paper) | https://pan.baidu.com/s/1e2NWX23QS_XdxgeszpB42Q (code: x7sf) | /mnt/forge-data/models/lsfdnet/net_LSFDNet.pth | MISSING |
| MMFusion branch | N/A | https://pan.baidu.com/s/1e2NWX23QS_XdxgeszpB42Q (code: x7sf) | /mnt/forge-data/models/lsfdnet/net_MMFusion.pth | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---------|------|-------|--------|------|--------|
| NSLSR (Nearshore Ship Long-Short Wave Registration) | 1,205 aligned SWIR/LWIR pairs | train/test = 844/361 (paper); train/val/test folders in repo README | https://pan.baidu.com/s/1Rm5w580LnY2JRYZvEJI0fw (code: shmp) | /mnt/forge-data/datasets/nslsr | MISSING |
| ISD | 1,045 pairs (paper text uses 940 train + 105 test for fusion eval) | train/test | ISD original publication source | /mnt/forge-data/datasets/isd | MISSING |

## Hyperparameters (from paper + released configs)
| Param | Value | Paper/Config Ref |
|-------|-------|---------------|
| optimizer (fusion and LSFDNet training) | Adam | Paper §4.1 |
| learning_rate | 1e-4 | Paper §4.1 |
| lr schedule | linear decay | Paper §4.1 |
| warmup_iter | 500 | Paper §4.1 |
| total_iter | 30,000 (paper), 30,000 in `LS_MMYOLO.yaml`, 300,000 in `LS_MMFusion.yaml` | Paper §4.1 + repo options |
| batch_size | 8 | Paper §4.1 |
| gamma correction (LWIR) | gamma in Eq. (6), config default 2.7 | Paper Eq. (6), `LS_MMYOLO.yaml` |
| OE loss coefficients | alpha=0.5, beta=0.5, sigma=0.2 in paper (repo defaults sigma=0.8) | Paper §4.1 + repo options |

## Expected Metrics (from paper and repo)
| Benchmark | Metric | Paper Value | Our Target |
|-----------|--------|-------------|-----------|
| NSLSR detection | mAP@0.5 | 0.962 (Table 2) | >=0.96 |
| NSLSR detection | mAP@0.5:0.95 | 0.770 (Table 2) | >=0.77 |
| NSLSR fusion | EN | 7.181 (Table 1, paper) | >=7.10 |
| NSLSR fusion | SF | 21.022 (Table 1, paper) | >=20.5 |
| NSLSR fusion | Qabf | 0.520 (Table 1, paper) | >=0.50 |

## Notes
- The repository README reports an updated checkpoint with higher detection metrics at 320x320 (`mAP@0.5 = 0.9855`, `mAP@0.5:0.95 = 0.7883`).
- This module should treat the paper tables as reproduction baseline and README updates as an improved target tier.
