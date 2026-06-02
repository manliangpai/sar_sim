#!/usr/bin/env python3
"""
生成各向同性正圆环的 28×28 格点金属块布局图（GT 可视化）。

圆环在物理坐标 (x,y) 上按 r=√(x²+y²) 定义，与仿真格心一致，各角度完全对称。
成像平面 0.28 m × 0.28 m，格距 0.01 m，Z=0.5 m，x/y ∈ [-0.14, +0.14] m。

运行（在 sar_sim 仓库根目录）::

  python test_code/generate_pattern_pics.py

输出 PNG 与 GT 数组 ``circle.npy``（28×28，uint8，1=金属块）。
默认写入 output/pics/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __name__ == "__main__" and __package__ is None:
    _repo = Path(__file__).resolve().parents[2]
    _pkg = Path(__file__).resolve().parents[1]
    for _root in (_repo, _pkg):
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from sar_sim.config.scene import PIXEL_CELL_M, PIXEL_GRID_N, PIXEL_PLANE_Z_M

PATTERN_NAME = "circle"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _PACKAGE_ROOT / "output" / "pics"

# 正圆环（物理坐标 m，圆心原点；R_in ≤ r ≤ R_out，各向同性）
CIRCLE_R_OUTER_M = 0.126
CIRCLE_R_INNER_M = 0.077

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


def validate_pattern(lines: Sequence[str], name: str) -> list[str]:
    if len(lines) != PIXEL_GRID_N:
        raise ValueError(f"{name}: 须 {PIXEL_GRID_N} 行，得到 {len(lines)}")
    out: list[str] = []
    for i, line in enumerate(lines):
        if len(line) != PIXEL_GRID_N:
            raise ValueError(f"{name} 行 {i}: 须 {PIXEL_GRID_N} 列，得到 {len(line)}")
        for ch in line:
            if ch not in "#.":
                raise ValueError(f"{name}: 非法字符 {ch!r}")
        out.append(line)
    return out


def _cell_center_m(col: int, display_row: int) -> tuple[float, float]:
    """与 config/scene._pixel_cell_center_m 相同的格心 (x, y)。"""
    n = PIXEL_GRID_N
    half = n * PIXEL_CELL_M * 0.5
    math_row = n - 1 - display_row
    x = -half + (col + 0.5) * PIXEL_CELL_M
    y = -half + (math_row + 0.5) * PIXEL_CELL_M
    return x, y


def _assert_circle_symmetry(mask: np.ndarray) -> None:
    """验证对 90° 旋转及镜像完全不变。"""
    if not (np.array_equal(mask, np.rot90(mask)) and np.array_equal(mask, mask.T)):
        raise RuntimeError("圆环未通过旋转/镜像对称性检查")
    if not (np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask))):
        raise RuntimeError("圆环未通过左右/上下镜像对称性检查")


def make_circle_pattern() -> list[str]:
    """
    正圆环：R_in ≤ √(x²+y²) ≤ R_out（m）。

    判定基于各格心物理坐标，与仿真散射体位置一致；在离散网格上
    对 90°/180°/270° 及镜像严格对称。
    """
    n = PIXEL_GRID_N
    lines: list[str] = []
    for display_row in range(n):
        row = []
        for col in range(n):
            x, y = _cell_center_m(col, display_row)
            r = float(np.hypot(x, y))
            row.append(
                "#"
                if CIRCLE_R_INNER_M <= r <= CIRCLE_R_OUTER_M
                else "."
            )
        lines.append("".join(row))
    pattern = validate_pattern(lines, PATTERN_NAME)
    _assert_circle_symmetry(pattern_to_array(pattern))
    return pattern


def pattern_to_array(pattern: list[str]) -> np.ndarray:
    """(N, N) uint8：1=金属格点，0=空。"""
    arr = np.zeros((PIXEL_GRID_N, PIXEL_GRID_N), dtype=np.uint8)
    for r, line in enumerate(pattern):
        for c, ch in enumerate(line):
            if ch == "#":
                arr[r, c] = 1
    return arr


def save_pattern_gt(pattern: list[str], path: Path) -> None:
    np.save(path, pattern_to_array(pattern))


def render_pattern_png(
    pattern: list[str],
    name: str,
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    """白底、黄色金属块、黑色网格线（网格在最上层）。"""
    mask = pattern_to_array(pattern)
    n_blocks = int(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, PIXEL_GRID_N)
    ax.set_ylim(0, PIXEL_GRID_N)
    ax.set_aspect("equal")

    for r in range(PIXEL_GRID_N):
        for c in range(PIXEL_GRID_N):
            if not mask[r, c]:
                continue
            y0 = PIXEL_GRID_N - 1 - r
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

    for i in range(PIXEL_GRID_N + 1):
        ax.plot(
            [0, PIXEL_GRID_N],
            [i, i],
            color="black",
            linewidth=0.6,
            zorder=10,
        )
        ax.plot(
            [i, i],
            [0, PIXEL_GRID_N],
            color="black",
            linewidth=0.6,
            zorder=10,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plane_m = PIXEL_GRID_N * PIXEL_CELL_M
    ax.set_title(
        f"{name}  ({n_blocks} blocks, {plane_m:.2f}×{plane_m:.2f} m, "
        f"cell={PIXEL_CELL_M} m, z={PIXEL_PLANE_Z_M} m)",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_circle(out_dir: Path, *, dpi: int = 120) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = make_circle_pattern()
    save_pattern_gt(pattern, out_dir / f"{PATTERN_NAME}.npy")
    render_pattern_png(pattern, PATTERN_NAME, out_dir / f"{PATTERN_NAME}.png", dpi=dpi)
    n_blocks = int(pattern_to_array(pattern).sum())
    print(f"  {PATTERN_NAME}: {n_blocks} blocks")

    keep = {PATTERN_NAME}
    for path in out_dir.glob("*.png"):
        if path.stem not in keep:
            path.unlink()
    for path in out_dir.glob("*.npy"):
        if path.stem not in keep:
            path.unlink()
    for path in out_dir.glob("*.txt"):
        path.unlink()
    for path in out_dir.glob("*.json"):
        path.unlink()

    return pattern


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成各向同性正圆环的 28×28 / 0.01 m / 0.28 m 平面 GT"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"输出目录（默认 {DEFAULT_OUT}）",
    )
    parser.add_argument("--dpi", type=int, default=120, help="PNG 分辨率")
    args = parser.parse_args()

    setup_chinese_font()
    plane_m = PIXEL_GRID_N * PIXEL_CELL_M
    print(
        f"生成图案 → {args.out.resolve()}  "
        f"({PIXEL_GRID_N}×{PIXEL_GRID_N}, {plane_m:.2f}×{plane_m:.2f} m, "
        f"cell={PIXEL_CELL_M} m, z={PIXEL_PLANE_Z_M} m)"
    )
    generate_circle(args.out, dpi=args.dpi)
    print("完成：1 个图案（circle）")


if __name__ == "__main__":
    main()
