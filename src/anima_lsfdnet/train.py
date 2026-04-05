from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .config import LSFDNetConfig, load_config
from .dataset import NSLSRDataset
from .losses import OEFusionLoss
from .model import create_model

ARTIFACTS = "/mnt/artifacts-datai"
PROJECT = "lsfdnet"


def _collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    swir = torch.stack([b["swir"] for b in batch])
    lwir = torch.stack([b["lwir"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"swir": swir, "lwir": lwir, "labels": labels}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WarmupLinearDecayLR:
    """Paper-faithful: linear warmup then linear decay to 0."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(0.0, 1.0 - progress)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSFDNet fusion core")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fallback", default=None)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override total_iter from config"
    )
    parser.add_argument(
        "--save-dir",
        default=f"{ARTIFACTS}/checkpoints/{PROJECT}",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def train(
    cfg: LSFDNetConfig, max_steps: int | None, save_dir: str | Path, resume: str | None
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"{ARTIFACTS}/logs/{PROJECT}")
    log_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.runtime.device
    _set_seed(cfg.train.seed)

    total_steps = max_steps if max_steps is not None else cfg.train.total_iter
    warmup_steps = min(cfg.train.warmup_iter, total_steps // 2)

    model = create_model(device)
    criterion = OEFusionLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = WarmupLinearDecayLR(optim, warmup_steps, total_steps)

    # Dataset with train/val split
    ds = NSLSRDataset(cfg.data.swir_dir, cfg.data.lwir_dir, cfg.data.label_dir or None)
    val_size = max(1, int(0.1 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.train.seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    start_step = 0
    best_val_loss = float("inf")

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[RESUME] from step {start_step}, best_val_loss={best_val_loss:.6f}")

    # Print training info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CONFIG] {cfg}")
    print(f"[MODEL] {n_params / 1e6:.2f}M parameters")
    print(f"[DATA] train={train_size}, val={val_size}")
    print(f"[TRAIN] {total_steps} steps, lr={cfg.train.lr}, warmup={warmup_steps}")
    print(f"[CKPT] saving to {save_dir}")

    model.train()
    step = start_step
    train_iter = iter(train_loader)
    epoch = 0
    running_loss = 0.0
    t0 = time.perf_counter()

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        swir = batch["swir"].to(device)
        lwir = batch["lwir"].to(device)
        labels_list: list[torch.Tensor] = batch["labels"]

        fused, _ = model(swir, lwir)

        # Compute loss per sample, average across batch
        losses = []
        for i in range(swir.shape[0]):
            sample_loss = criterion(
                swir=swir[i : i + 1],
                lwir=lwir[i : i + 1],
                fused=fused[i : i + 1],
                labels=labels_list[i].to(device),
                alpha=cfg.train.alpha,
                beta=cfg.train.beta,
                sigma=cfg.train.sigma,
                gamma=cfg.train.gamma,
            )
            losses.append(sample_loss)
        loss = torch.stack(losses).mean()

        if torch.isnan(loss):
            print(f"[FATAL] NaN loss at step {step}. Stopping.")
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        step += 1
        running_loss += loss.item()

        # Log every 100 steps
        if step % 100 == 0:
            avg = running_loss / 100
            lr = optim.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t0
            print(f"[step {step}/{total_steps}] loss={avg:.6f} lr={lr:.2e} elapsed={elapsed:.0f}s")
            running_loss = 0.0

        # Validate and checkpoint every 500 steps
        if step % 500 == 0 or step == total_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vswir = vbatch["swir"].to(device)
                    vlwir = vbatch["lwir"].to(device)
                    vlabels: list[torch.Tensor] = vbatch["labels"]
                    vfused, _ = model(vswir, vlwir)
                    for i in range(vswir.shape[0]):
                        vl = criterion(
                            swir=vswir[i : i + 1],
                            lwir=vlwir[i : i + 1],
                            fused=vfused[i : i + 1],
                            labels=vlabels[i].to(device),
                            alpha=cfg.train.alpha,
                            beta=cfg.train.beta,
                            sigma=cfg.train.sigma,
                            gamma=cfg.train.gamma,
                        )
                        val_losses.append(vl.item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
            print(f"[VAL step {step}] val_loss={val_loss:.6f} (best={best_val_loss:.6f})")

            ckpt_state = {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": min(best_val_loss, val_loss),
                "config": str(cfg),
            }

            # Save latest
            torch.save(ckpt_state, save_dir / f"checkpoint_step{step:06d}.pth")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt_state, save_dir / "best.pth")
                print(f"[CKPT] New best: val_loss={val_loss:.6f}")

            # Cleanup old checkpoints (keep best + last 2)
            ckpts = sorted(save_dir.glob("checkpoint_step*.pth"))
            while len(ckpts) > 2:
                ckpts[0].unlink()
                ckpts.pop(0)

            model.train()

    print(f"[DONE] Training complete. Best val_loss={best_val_loss:.6f}")
    print(f"[CKPT] Best model: {save_dir / 'best.pth'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.fallback)
    train(cfg=cfg, max_steps=args.max_steps, save_dir=args.save_dir, resume=args.resume)


if __name__ == "__main__":
    main()
