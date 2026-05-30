#!/usr/bin/env python3
"""
训练 Baseline UNet：processed BP magnitude → GT 二值掩膜。

运行（在 sar_sim 仓库根目录）::

  python train/scripts/train.py
  python train/scripts/train.py --config train/configs/baseline_unet.json --epochs 300
  python train/scripts/train.py --gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parents[2].parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader

from sar_sim.train.datasets.pixel_pattern import (
    DataSource,
    PixelPatternDataset,
    list_paired_samples,
    sources_from_config,
    split_samples_random,
)
from sar_sim.train.losses.segmentation import segmentation_loss
from sar_sim.train.metrics.segmentation import segmentation_metrics
from sar_sim.train.models.unet import UNetSmall
from sar_sim.train.utils.io import ensure_dir, get_device, seed_everything

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else _PACKAGE_ROOT / p


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"iou": 0.0, "dice": 0.0, "acc": 0.0, "loss": 0.0}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        loss, _ = segmentation_loss(logits, y)
        m = segmentation_metrics(logits, y)
        totals["loss"] += float(loss.item())
        totals["iou"] += m["iou"]
        totals["dice"] += m["dice"]
        totals["acc"] += m["acc"]
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, parts = segmentation_loss(logits, y)
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["bce"] += parts["bce"]
        totals["dice"] += parts["dice"]
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def build_sources(cfg: dict) -> list[DataSource]:
    return sources_from_config(cfg, _PACKAGE_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 UNet baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PACKAGE_ROOT / "train/configs/baseline_unet.json",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sources = build_sources(cfg)
    val_ratio = float(cfg.get("val_ratio", 0.3))
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    lr = args.lr if args.lr is not None else float(train_cfg["lr"])
    seed = int(train_cfg.get("seed", 42))
    log1p = bool(train_cfg.get("log1p", True))

    ckpt_dir = ensure_dir(resolve_path(cfg["checkpoint_dir"]))
    runs_dir = ensure_dir(resolve_path(cfg["runs_dir"]))

    seed_everything(seed)
    device = get_device(prefer_cuda=args.gpu and not args.cpu)

    all_samples = list_paired_samples(sources)
    train_samples, val_samples = split_samples_random(
        all_samples, val_ratio=val_ratio, seed=seed
    )
    val_names = sorted(s.name for s in val_samples)
    print(f"train: {len(train_samples)}  val: {len(val_samples)} (val_ratio={val_ratio})")
    print(f"  val: {val_names}")
    for src in sources:
        print(f"  source: {src.processed_dir.name} ← {src.gt_dir.name}")
    print(f"device: {device}")

    train_ds = PixelPatternDataset(samples=train_samples, log1p=log1p)
    val_ds = PixelPatternDataset(samples=val_samples, log1p=log1p)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNetSmall(base_channels=int(model_cfg.get("base_channels", 32))).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    best_iou = -1.0
    history: list[dict[str, float | int]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:3d}/{epochs}  "
                f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
                f"val_iou={va['iou']:.4f}  val_dice={va['dice']:.4f}",
                flush=True,
            )

        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "config": cfg,
                    "train_names": [s.name for s in train_samples],
                    "val_names": [s.name for s in val_samples],
                },
                ckpt_dir / "best.pt",
            )

    elapsed = time.perf_counter() - t0
    last_path = ckpt_dir / "last.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs, "config": cfg}, last_path)

    meta = {
        "elapsed_s": round(elapsed, 1),
        "best_val_iou": best_iou,
        "train_names": [s.name for s in train_samples],
        "val_names": [s.name for s in val_samples],
        "history": history,
    }
    (runs_dir / "train_history.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"done in {elapsed:.1f}s  best_val_iou={best_iou:.4f}")
    print(f"checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
