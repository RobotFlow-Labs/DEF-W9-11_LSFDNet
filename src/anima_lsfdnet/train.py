from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import LSFDNetConfig, load_config
from .dataset import NSLSRDataset
from .losses import OEFusionLoss
from .model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSFDNet fusion core")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fallback", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save", default="checkpoints/lsfdnet_fusion_core.pth")
    return parser.parse_args()


def train(cfg: LSFDNetConfig, epochs: int, save_path: str | Path) -> None:
    device = cfg.runtime.device
    model = create_model(device)
    criterion = OEFusionLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    ds = NSLSRDataset(cfg.data.swir_dir, cfg.data.lwir_dir, cfg.data.label_dir or None)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.runtime.num_workers)

    model.train()
    for epoch in range(epochs):
        for batch in loader:
            swir = batch["swir"].to(device)
            lwir = batch["lwir"].to(device)
            labels = batch["labels"]
            if labels.ndim == 3:
                labels = labels[0]
            labels = labels.to(device)

            fused, _ = model(swir, lwir)
            loss = criterion(
                swir=swir,
                lwir=lwir,
                fused=fused,
                labels=labels,
                alpha=cfg.train.alpha,
                beta=cfg.train.beta,
                sigma=cfg.train.sigma,
                gamma=cfg.train.gamma,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"saved checkpoint: {save_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.fallback)
    train(cfg=cfg, epochs=args.epochs, save_path=args.save)


if __name__ == "__main__":
    main()
