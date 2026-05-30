"""processed BP magnitude (20×20) ↔ pics GT npy (20×20) 配对 Dataset。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sar_sim.config.scene import DEFAULT_PATTERN_PICS_DIR, load_pixel_pattern_gt

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = _PACKAGE_ROOT / "output" / "processed_radar_data_z0.6"
DEFAULT_GT_DIR = DEFAULT_PATTERN_PICS_DIR


@dataclass(frozen=True)
class DataSource:
    processed_dir: Path
    gt_dir: Path


@dataclass(frozen=True)
class PatternSample:
    name: str
    processed_dir: Path
    gt_dir: Path


def list_paired_pattern_names(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    gt_dir: Path = DEFAULT_GT_DIR,
) -> list[str]:
    """同时存在 processed npz 与 GT npy 的图案名（单数据源）。"""
    proc = {p.stem for p in processed_dir.glob("*.npz")}
    gt = {p.stem for p in gt_dir.glob("*.npy")}
    return sorted(proc & gt)


def list_paired_samples(sources: Sequence[DataSource]) -> list[PatternSample]:
    """多数据源：返回 (name, processed_dir, gt_dir) 列表。"""
    samples: list[PatternSample] = []
    seen: set[str] = set()
    for src in sources:
        proc_dir = Path(src.processed_dir)
        gt_dir = Path(src.gt_dir)
        for name in list_paired_pattern_names(proc_dir, gt_dir):
            if name in seen:
                continue
            seen.add(name)
            samples.append(
                PatternSample(name=name, processed_dir=proc_dir, gt_dir=gt_dir)
            )
    return sorted(samples, key=lambda s: s.name)


def sources_from_config(cfg: dict, root: Path) -> list[DataSource]:
    if "sources" in cfg:
        return [
            DataSource(
                processed_dir=root / s["processed_dir"],
                gt_dir=root / s["gt_dir"],
            )
            for s in cfg["sources"]
        ]
    return [
        DataSource(
            processed_dir=root / cfg["processed_dir"],
            gt_dir=root / cfg["gt_dir"],
        )
    ]


def split_pattern_names(
    names: Sequence[str],
    val_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    val_set = set(val_names)
    train = [n for n in names if n not in val_set]
    val = [n for n in names if n in val_set]
    if not train:
        raise ValueError("训练集为空，请调整 val_names")
    if not val:
        raise ValueError("验证集为空，请调整 val_names")
    return train, val


def split_samples(
    samples: Sequence[PatternSample],
    val_names: Sequence[str],
) -> tuple[list[PatternSample], list[PatternSample]]:
    val_set = set(val_names)
    train = [s for s in samples if s.name not in val_set]
    val = [s for s in samples if s.name in val_set]
    if not train:
        raise ValueError("训练集为空，请调整 val_names")
    if not val:
        raise ValueError("验证集为空，请调整 val_names")
    return train, val


def split_samples_random(
    samples: Sequence[PatternSample],
    *,
    val_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[PatternSample], list[PatternSample]]:
    """按 val_ratio 随机划分；至少保留 1 个训练样本与 1 个验证样本。"""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 须在 (0,1) 内，得到 {val_ratio}")
    items = sorted(samples, key=lambda s: s.name)
    if len(items) < 2:
        raise ValueError("至少需要 2 个配对样本才能划分训练/验证")
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, round(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)
    val_idx = set(order[:n_val])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]
    return train, val


def resolve_val_samples(
    cfg: dict,
    root: Path,
    *,
    val_names: Sequence[str] | None = None,
    seed: int | None = None,
) -> list[PatternSample]:
    """从配置与 checkpoint 信息解析验证样本列表。"""
    all_samples = list_paired_samples(sources_from_config(cfg, root))
    if val_names:
        val_set = set(val_names)
        val = [s for s in all_samples if s.name in val_set]
        if not val:
            raise ValueError(f"验证名在数据中未找到: {val_names}")
        return val
    val_ratio = float(cfg.get("val_ratio", 0.3))
    split_seed = seed if seed is not None else int(cfg.get("train", {}).get("seed", 42))
    _, val = split_samples_random(all_samples, val_ratio=val_ratio, seed=split_seed)
    return val


def normalize_magnitude(mag: np.ndarray, *, log1p: bool = True) -> np.ndarray:
    """单样本归一化 → float32。"""
    x = mag.astype(np.float32, copy=False)
    if log1p:
        x = np.log1p(x)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return x - mean
    return (x - mean) / std


class PixelPatternDataset(Dataset):
    def __init__(
        self,
        pattern_names: Sequence[str] | None = None,
        *,
        samples: Sequence[PatternSample] | None = None,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        gt_dir: Path = DEFAULT_GT_DIR,
        log1p: bool = True,
    ) -> None:
        if samples is not None:
            self._samples = list(samples)
        elif pattern_names is not None:
            self._samples = [
                PatternSample(
                    name=n,
                    processed_dir=Path(processed_dir),
                    gt_dir=Path(gt_dir),
                )
                for n in pattern_names
            ]
        else:
            raise ValueError("须指定 pattern_names 或 samples")

        self.log1p = log1p
        for s in self._samples:
            if not (s.processed_dir / f"{s.name}.npz").is_file():
                raise FileNotFoundError(s.processed_dir / f"{s.name}.npz")
            if not (s.gt_dir / f"{s.name}.npy").is_file():
                raise FileNotFoundError(s.gt_dir / f"{s.name}.npy")

    @property
    def pattern_names(self) -> list[str]:
        return [s.name for s in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self._samples[index]
        proc = np.load(sample.processed_dir / f"{sample.name}.npz")
        mag = proc["magnitude"]
        gt = load_pixel_pattern_gt(sample.gt_dir / f"{sample.name}.npy")

        x = normalize_magnitude(mag, log1p=self.log1p)
        x_t = torch.from_numpy(x).unsqueeze(0)
        y_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)

        return {"x": x_t, "y": y_t, "name": sample.name}
