# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 55%

## Done
- [x] Paper analyzed (arXiv 2507.20574 -- LSFDNet SWIR/LWIR fusion + detection)
- [x] ASSETS.md with weights, datasets, hyperparameters, and expected metrics
- [x] 7 PRDs covering foundation through production hardening
- [x] 21 task files in tasks/ (3 per PRD)
- [x] Config system with TOML loading and dataclasses (config.py)
- [x] Dataset loader for aligned SWIR/LWIR pairs (dataset.py)
- [x] Core fusion model: MTFE + MLCF + Decoder (model.py, blocks.py)
- [x] OE fusion loss implementation matching Eq. 5-14 (losses.py)
- [x] Training script with LR scheduler, warmup, grad clipping, validation (train.py)
- [x] Inference CLI pipeline (infer.py, io.py)
- [x] Evaluation runner with VIF metric (eval.py, metrics.py)
- [x] FastAPI serving endpoints (api.py)
- [x] AnimaNode serve.py for Docker serving
- [x] Export pipeline pth->safetensors->ONNX->TRT (export.py)
- [x] Monitoring/observability hooks (monitoring.py)
- [x] Asset validation utility (checks.py)
- [x] Unit tests for config, dataset, model, inference, metrics, API (10/10 passing)
- [x] Docker serving files (Dockerfile.serve, docker-compose.serve.yml)
- [x] ROS2 node and launch file scaffolds
- [x] anima_module.yaml manifest
- [x] pyproject.toml with hatchling build + cu128 index
- [x] CLAUDE.md (paper summary, architecture, hyperparams)
- [x] PRD.md (master build plan with 7 PRDs table)
- [x] debug.toml config for smoke testing
- [x] .venv created on /mnt/artifacts-datai/venvs/lsfdnet (symlinked)
- [x] All deps installed: torch 2.11.0+cu128, 8x L4 verified
- [x] Code review: fixed 9 issues (sigma inversion, separate base extractors, decoder arch, separate FFNs, training loop, VIF metric, serve.py)
- [x] ruff lint: 0 errors
- [x] pytest: 10/10 passing

## In Progress
- [ ] Waiting for NSLSR/ISD datasets and pretrained weights (Baidu Pan)

## TODO
- [ ] Download NSLSR dataset from Baidu Pan -- code: shmp (MUST download in China, Baidu Pan)
- [ ] Download ISD dataset from Baidu Pan (MUST download in China, Baidu Pan)
- [ ] Download pretrained weights from Baidu Pan -- code: x7sf (MUST download in China, Baidu Pan)
- [ ] Real-data training of fusion core on NSLSR (GPU batch finder + nohup)
- [ ] Integrate YOLOv12 detection branch from reference repo (MMYOLO)
- [ ] End-to-end joint training (fusion + detection)
- [ ] Reproduce paper metrics (Table 1 fusion, Table 2 detection)
- [ ] ONNX/TensorRT export with real weights
- [ ] Push trained weights to HuggingFace (ilessio-aiflowlab/project_lsfdnet)
- [ ] Full ROS2 node implementation (requires rclpy environment)
- [ ] Generate TRAINING_REPORT.md with final metrics

## Blocking
- NSLSR and ISD datasets are on Baidu Pan -- REQUIRES DOWNLOAD IN CHINA (Baidu Pan blocked outside China), then scp to server
- Pretrained weights are on Baidu Pan -- same, must download in China
- /mnt/forge-data disk is 100% full (0 bytes free) -- .venv created on artifacts disk as workaround
- UV cache redirected: /home/datai/.cache/uv -> /mnt/artifacts-datai/cache/uv

## Downloads Needed
- NSLSR dataset (~1,205 image pairs) -- Baidu Pan code: shmp -- scp to /mnt/forge-data/datasets/nslsr
- ISD dataset (~1,045 image pairs) -- Baidu Pan (see original publication) -- scp to /mnt/forge-data/datasets/isd
- LSFDNet weights -- Baidu Pan code: x7sf -- scp to /mnt/forge-data/models/lsfdnet/

## Code Review Fixes Applied (2026-04-05)
1. losses.py:82 -- sigma weighting inverted vs paper Eq. 14 (was global*sigma, now global*(1-sigma))
2. model.py:103 -- shared FeatureExtractorBase -> separate base_sw/base_lw per paper
3. model.py:75 -- FusionDecoder had 8 layers, reduced to 4+1 matching paper Section 3.3
4. blocks.py:76-77 -- same FFN reused for self-attn and cross-attn -> separate ffn_s_cross/ffn_l_cross
5. train.py -- complete rewrite: iteration-based, LR warmup+linear decay, grad clipping, validation loop, custom collate, checkpoint to /mnt/artifacts-datai/
6. metrics.py -- added VIF metric (paper Table 1 target >= 0.60)
7. eval.py -- added VIF to reporting and paper baseline
8. serve.py -- created AnimaNode-compatible LSFDNetNode class
9. export.py -- added safetensors + TRT export support
