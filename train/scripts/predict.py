#!/usr/bin/env python3
"""
单图案推理并保存三联图：GT | BP | Prediction。

运行::

  python train/scripts/predict.py --pattern circle
  python train/scripts/predict.py --pattern 8 --gpu
  python train/scripts/predict.py --val --gpu
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

import numpy as np
import torch

from sar_sim.config.scene import load_pixel_pattern_gt
from sar_sim.train.datasets.pixel_pattern import (
    PatternSample,
    resolve_val_samples,
    sources_from_config,
    normalize_magnitude,
)
from sar_sim.train.models.unet import UNetSmall
from sar_sim.train.utils.io import get_device
from sar_sim.train.utils.viz import logits_to_prob, save_triplet_figure

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def predict_one(
    pattern: str,
    model: torch.nn.Module,
    device: torch.device,
    *,
    processed_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    log1p: bool,
) -> Path:
    proc_path = processed_dir / f"{pattern}.npz"
    gt_path = gt_dir / f"{pattern}.npy"
    if not proc_path.is_file():
        raise FileNotFoundError(f"缺少 processed: {proc_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 GT: {gt_path}")

    proc = np.load(proc_path)
    mag = proc["magnitude"]
    gt = load_pixel_pattern_gt(gt_path)
    x = normalize_magnitude(mag, log1p=log1p)
    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_t.to(device))
    prob = logits_to_prob(logits)

    bp_vis = mag.astype(np.float32)
    bp_vis = bp_vis / (bp_vis.max() + 1e-6)

    out_path = out_dir / f"{pattern}_triplet.png"
    save_triplet_figure(bp_vis, gt.astype(np.float32), prob, name=pattern, out_path=out_path)
    return out_path


def default_source(cfg: dict) -> tuple[Path, Path]:
    sources = sources_from_config(cfg, _PACKAGE_ROOT)
    src = sources[0]
    return src.processed_dir, src.gt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="图案预测与三联图可视化")
    parser.add_argument(
        "--pattern",
        nargs="*",
        default=None,
        metavar="NAME",
        help="图案名（可多个）；默认 circle",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="对 checkpoint 中记录的验证集全部出图",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_PACKAGE_ROOT / "train/checkpoints/best.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PACKAGE_ROOT / "train/runs/predictions",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"未找到 checkpoint: {args.checkpoint}，请先运行 train.py")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    log1p = bool(cfg.get("train", {}).get("log1p", True))
    base_ch = int(cfg.get("model", {}).get("base_channels", 32))

    if args.val:
        jobs = resolve_val_samples(cfg, _PACKAGE_ROOT, val_names=ckpt.get("val_names"))
    elif args.pattern:
        processed_dir, gt_dir = default_source(cfg)
        jobs = [
            PatternSample(name=n, processed_dir=processed_dir, gt_dir=gt_dir)
            for n in args.pattern
        ]
    else:
        processed_dir, gt_dir = default_source(cfg)
        jobs = [
            PatternSample(name="circle", processed_dir=processed_dir, gt_dir=gt_dir)
        ]

    device = get_device(prefer_cuda=args.gpu and not args.cpu)
    model = UNetSmall(base_channels=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        out_path = predict_one(
            job.name,
            model,
            device,
            processed_dir=job.processed_dir,
            gt_dir=job.gt_dir,
            out_dir=args.out_dir,
            log1p=log1p,
        )
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
