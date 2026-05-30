#!/usr/bin/env python3
"""
在验证集上评估 checkpoint。

运行::

  python train/scripts/eval.py
  python train/scripts/eval.py --checkpoint train/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parents[2].parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader

from sar_sim.train.datasets.pixel_pattern import PixelPatternDataset, resolve_val_samples
from sar_sim.train.metrics.segmentation import segmentation_metrics
from sar_sim.train.models.unet import UNetSmall
from sar_sim.train.utils.io import get_device

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 UNet checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_PACKAGE_ROOT / "train/checkpoints/best.pt",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"未找到 checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    log1p = bool(cfg.get("train", {}).get("log1p", True))
    base_ch = int(cfg.get("model", {}).get("base_channels", 32))

    val_samples = resolve_val_samples(
        cfg,
        _PACKAGE_ROOT,
        val_names=ckpt.get("val_names"),
    )
    ds = PixelPatternDataset(samples=val_samples, log1p=log1p)

    device = get_device(prefer_cuda=args.gpu and not args.cpu)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNetSmall(base_channels=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"eval {len(ds)} patterns on {device}")
    totals = {"iou": 0.0, "dice": 0.0, "acc": 0.0}
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        name = batch["name"]
        if isinstance(name, (list, tuple)):
            name = name[0]
        with torch.no_grad():
            logits = model(x)
        m = segmentation_metrics(logits, y)
        totals["iou"] += m["iou"]
        totals["dice"] += m["dice"]
        totals["acc"] += m["acc"]
        print(f"  {name}: IoU={m['iou']:.4f}  Dice={m['dice']:.4f}  Acc={m['acc']:.4f}")

    n = len(ds)
    print(
        f"mean: IoU={totals['iou']/n:.4f}  "
        f"Dice={totals['dice']/n:.4f}  Acc={totals['acc']/n:.4f}"
    )


if __name__ == "__main__":
    main()
