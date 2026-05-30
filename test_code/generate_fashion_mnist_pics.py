#!/usr/bin/env python3
"""
Fashion-MNIST → 20×20 二值 GT（.npy + 预览 .png），用于 SAR 仿真。

处理流程（与预览对比图一致）::

  28×28 原图 → 固定阈值二值化 → 最近邻等比例缩小至 20×20

运行（在 sar_sim 仓库根目录）::

  python test_code/generate_fashion_mnist_pics.py
  python test_code/generate_fashion_mnist_pics.py --per-class 2 --seed 42
  python test_code/generate_fashion_mnist_pics.py --compare
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image

SRC_GRID_N = 28
OUT_GRID_N = 20
THRESHOLD = 10

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _PACKAGE_ROOT / "output" / "pics"
CACHE_DIR = _PACKAGE_ROOT / ".cache" / "fashion_mnist"

FMNIST_MIRRORS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
)
FMNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
}

CLASS_NAMES = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
)


def setup_chinese_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def download_fashion_mnist(cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, fname in FMNIST_FILES.items():
        dest = cache_dir / fname
        if not dest.is_file():
            last_err: Exception | None = None
            for base in FMNIST_MIRRORS:
                url = f"{base}/{fname}"
                try:
                    print(f"download {url} ...", flush=True)
                    urllib.request.urlretrieve(url, dest)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    if dest.is_file():
                        dest.unlink()
            if last_err is not None:
                raise RuntimeError(f"无法下载 {fname}: {last_err}") from last_err
        paths[key] = dest
    return paths


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic {magic} in {path}")
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic {magic} in {path}")
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def binarize(img: np.ndarray, *, threshold: int = THRESHOLD) -> np.ndarray:
    """黑底亮物体 → uint8 掩膜，1=前景。"""
    return (img >= threshold).astype(np.uint8)


def downscale_binary(binary: np.ndarray, out_size: int = OUT_GRID_N) -> np.ndarray:
    """二值掩膜等比例缩小（最近邻，保持 0/1 形状）。"""
    h, w = binary.shape[:2]
    if h != w:
        raise ValueError(f"须为正方形，得到 {w}×{h}")
    pil = Image.fromarray(binary.astype(np.uint8) * 255)
    small = pil.resize((out_size, out_size), Image.Resampling.NEAREST)
    return (np.asarray(small) >= 128).astype(np.uint8)


def image_to_gt(img: np.ndarray, *, threshold: int = THRESHOLD) -> np.ndarray:
    """28×28 灰度图 → (20, 20) uint8 GT。"""
    gt = downscale_binary(binarize(img, threshold=threshold))
    if not np.any(gt):
        raise ValueError("二值化并缩小后 GT 全空")
    if not np.all(np.isin(gt, (0, 1))):
        raise ValueError("GT 须仅含 0/1")
    return gt


def pick_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    seed: int | None,
) -> list[tuple[str, int, int, np.ndarray]]:
    """每类随机 per_class 张，返回 (类名, slot, 源索引, 28×28 图)。"""
    rng = np.random.default_rng(seed)
    entries: list[tuple[str, int, int, np.ndarray]] = []
    for cls, class_name in enumerate(CLASS_NAMES):
        pool = np.flatnonzero(labels == cls)
        if len(pool) < per_class:
            raise ValueError(
                f"{class_name}: 仅 {len(pool)} 张，需要 {per_class} 张"
            )
        chosen = rng.choice(pool, size=per_class, replace=False)
        for slot, idx in enumerate(sorted(int(i) for i in chosen), 1):
            entries.append((class_name, slot, idx, images[idx]))
    return entries


def render_gt_png(
    mask: np.ndarray,
    name: str,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    """白底、黄块、黑网格（与 generate_pattern_pics 一致）。"""
    n = OUT_GRID_N
    n_blocks = int(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")

    for r in range(n):
        for c in range(n):
            if not mask[r, c]:
                continue
            y0 = n - 1 - r
            ax.add_patch(
                Rectangle(
                    (c, y0),
                    1,
                    1,
                    facecolor="#FFD700",
                    edgecolor="none",
                    zorder=1,
                )
            )

    for i in range(n + 1):
        ax.plot([0, n], [i, i], color="black", linewidth=1.0, zorder=10)
        ax.plot([i, i], [0, n], color="black", linewidth=1.0, zorder=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{name}  ({n_blocks} blocks)", fontsize=13)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def draw_grid(ax: plt.Axes, n: int) -> None:
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    for i in range(n + 1):
        ax.plot([0, n], [i, i], color="black", linewidth=0.6, zorder=10)
        ax.plot([i, i], [0, n], color="black", linewidth=0.6, zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_compare_figure(
    img: np.ndarray,
    binary: np.ndarray,
    gt: np.ndarray,
    *,
    title: str,
    path: Path,
    threshold: int,
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    fig.patch.set_facecolor("white")

    axes[0].imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[0, SRC_GRID_N, SRC_GRID_N, 0],
        interpolation="nearest",
        zorder=1,
    )
    draw_grid(axes[0], SRC_GRID_N)
    axes[0].set_title("原图 28×28", fontsize=12)

    axes[1].imshow(
        binary,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, SRC_GRID_N, SRC_GRID_N, 0],
        interpolation="nearest",
        zorder=1,
    )
    draw_grid(axes[1], SRC_GRID_N)
    axes[1].set_title(f"二值化 28×28 (≥{threshold})", fontsize=12)

    axes[2].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, OUT_GRID_N, OUT_GRID_N, 0],
        interpolation="nearest",
        zorder=1,
    )
    draw_grid(axes[2], OUT_GRID_N)
    axes[2].set_title("二值化后等比例缩小 20×20", fontsize=12)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def cleanup_stale(out_dir: Path, keep: set[str], *, keep_compare: bool) -> None:
    """删除 out_dir 根目录下旧 GT。"""
    for path in out_dir.glob("*.png"):
        stem = path.stem
        if stem.endswith("_compare"):
            base = stem[: -len("_compare")]
            if not keep_compare or base not in keep:
                path.unlink(missing_ok=True)
        elif stem not in keep:
            path.unlink(missing_ok=True)
    for path in out_dir.glob("*.npy"):
        if path.stem not in keep:
            path.unlink(missing_ok=True)


def export_all(
    out_dir: Path,
    *,
    per_class: int = 2,
    seed: int = 42,
    threshold: int = THRESHOLD,
    dpi: int = 150,
    compare: bool = False,
) -> list[str]:
    paths = download_fashion_mnist()
    images = read_idx_images(paths["train_images"])
    labels = read_idx_labels(paths["train_labels"])

    entries = pick_per_class(images, labels, per_class=per_class, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    print(
        f"export: {per_class}/class × {len(CLASS_NAMES)} = "
        f"{len(entries)} → {out_dir.resolve()}"
    )

    for class_name, slot, src_idx, img in entries:
        name = f"{class_name}_{slot}"
        binary = binarize(img, threshold=threshold)
        gt = image_to_gt(img, threshold=threshold)

        np.save(out_dir / f"{name}.npy", gt)
        render_gt_png(gt, name, out_dir / f"{name}.png", dpi=dpi)

        if compare:
            save_compare_figure(
                img,
                binary,
                gt,
                title=f"{name}  (src idx={src_idx})",
                path=out_dir / f"{name}_compare.png",
                threshold=threshold,
                dpi=dpi,
            )

        names.append(name)
        print(f"  {name}: blocks={int(gt.sum())}  src_idx={src_idx}", flush=True)

    cleanup_stale(out_dir, set(names), keep_compare=compare)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST → 20×20 GT（output/pics）"
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=2,
        help="每类随机采样张数（默认 2，共 20 个 GT）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--threshold",
        type=int,
        default=THRESHOLD,
        help=f"二值化阈值 pixel >= N（默认 {THRESHOLD}）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"输出目录（默认 {DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="额外输出三联对比图 {name}_compare.png",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.per_class < 1:
        raise SystemExit("per-class 须 >= 1")

    setup_chinese_font()
    names = export_all(
        args.out,
        per_class=args.per_class,
        seed=args.seed,
        threshold=args.threshold,
        dpi=args.dpi,
        compare=args.compare,
    )
    print(f"done: {len(names)} patterns")


if __name__ == "__main__":
    main()
