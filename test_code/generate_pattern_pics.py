#!/usr/bin/env python3
"""
生成 20×20、1 cm 格点金属块布局图（GT 可视化）。

成像平面：20 cm × 20 cm，格距 1 cm，对应 Z=0.6 m 平面上 1 cm 不锈钢方块。
「#」= 有金属，「.」= 空。

运行（在 sar_sim 仓库根目录）::

  python test_code/generate_pattern_pics.py
  python test_code/generate_pattern_pics.py

输出 PNG（白底、黄块、黑色网格线置顶）与 GT 数组 ``{NAME}.npy``（20×20，uint8，1=金属块）。
默认写入 output/pics/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

GRID_N = 20
CELL_M = 0.01
PLANE_Z_M = 0.6
PLANE_SIZE_M = GRID_N * CELL_M

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _PACKAGE_ROOT / "output" / "pics"

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


def blank_grid() -> list[str]:
    return ["." * GRID_N for _ in range(GRID_N)]


def validate_pattern(lines: Sequence[str], name: str) -> list[str]:
    if len(lines) != GRID_N:
        raise ValueError(f"{name}: 须 {GRID_N} 行，得到 {len(lines)}")
    out: list[str] = []
    for i, line in enumerate(lines):
        if len(line) != GRID_N:
            raise ValueError(f"{name} 行 {i}: 须 {GRID_N} 列，得到 {len(line)}")
        for ch in line:
            if ch not in "#.":
                raise ValueError(f"{name}: 非法字符 {ch!r}")
        out.append(line)
    return out


def embed_pattern(sub: list[str], canvas: list[str] | None = None) -> list[str]:
    """将小图案居中嵌入 20×20。"""
    h = len(sub)
    w = len(sub[0]) if sub else 0
    if h > GRID_N or w > GRID_N:
        raise ValueError(f"子图案 {w}×{h} 超出 {GRID_N}×{GRID_N}")
    base = [list(row) for row in (canvas or blank_grid())]
    r0 = (GRID_N - h) // 2
    c0 = (GRID_N - w) // 2
    for r, line in enumerate(sub):
        for c, ch in enumerate(line):
            if ch == "#":
                base[r0 + r][c0 + c] = "#"
    return ["".join(row) for row in base]


def scale_pattern(pattern: list[str], factor: int) -> list[str]:
    out: list[str] = []
    for line in pattern:
        row: list[str] = []
        for ch in line:
            row.extend([ch] * factor)
        for _ in range(factor):
            out.append("".join(row))
    return out


def draw_rect(
    canvas: list[list[str]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> None:
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if 0 <= r < GRID_N and 0 <= c < GRID_N:
                canvas[r][c] = "#"


def pattern_to_array(pattern: list[str]) -> np.ndarray:
    """(20, 20) uint8：1=金属格点，0=空。"""
    arr = np.zeros((GRID_N, GRID_N), dtype=np.uint8)
    for r, line in enumerate(pattern):
        for c, ch in enumerate(line):
            if ch == "#":
                arr[r, c] = 1
    return arr


# ---------------------------------------------------------------------------
# LED 7 段：数字 0–9
# ---------------------------------------------------------------------------

_SEG_NAMES = ("a", "b", "c", "d", "e", "f", "g")
_SEG_DIGITS: dict[str, tuple[int, ...]] = {
    "0": (0, 1, 2, 3, 4, 5),
    "1": (1, 2),
    "2": (0, 1, 3, 4, 6),
    "3": (0, 1, 2, 3, 6),
    "4": (1, 2, 5, 6),
    "5": (0, 2, 3, 5, 6),
    "6": (0, 2, 3, 4, 5, 6),
    "7": (0, 1, 2),
    "8": (0, 1, 2, 3, 4, 5, 6),
    "9": (0, 1, 2, 3, 5, 6),
}


def _seg7_rects(thickness: int = 2) -> dict[str, tuple[int, int, int, int]]:
    t = thickness
    c0, r0 = 4, 2
    cw, ch = 12, 16
    return {
        "a": (r0, c0 + 2, t, cw - 4),
        "g": (r0 + ch // 2 - t // 2, c0 + 2, t, cw - 4),
        "d": (r0 + ch - t, c0 + 2, t, cw - 4),
        "f": (r0 + t, c0, ch // 2 - t, t),
        "b": (r0 + t, c0 + cw - t, ch // 2 - t, t),
        "e": (r0 + ch // 2, c0, ch // 2 - t, t),
        "c": (r0 + ch // 2, c0 + cw - t, ch // 2 - t, t),
    }


def make_seg7_digit(d: str) -> list[str]:
    if d not in _SEG_DIGITS:
        raise ValueError(d)
    seg_rects = _seg7_rects(thickness=2)
    g = [list("." * GRID_N) for _ in range(GRID_N)]
    on = set(_SEG_DIGITS[d])
    for i, name in enumerate(_SEG_NAMES):
        if i in on:
            r0, c0, h, w = seg_rects[name]
            draw_rect(g, r0, c0, h, w)
    return ["".join(row) for row in g]


# ---------------------------------------------------------------------------
# LED 风格字母：5×7 点阵 ×2 放大
# ---------------------------------------------------------------------------

_FONT_5X7: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def make_led_letter(ch: str) -> list[str]:
    raw = _FONT_5X7[ch.upper()]
    sub: list[str] = []
    for line in raw:
        sub.append("".join("#" if c == "1" else "." for c in line))
    scaled = scale_pattern(sub, factor=2)
    return embed_pattern(scaled)


def collect_all_patterns() -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {}
    for d in "0123456789":
        patterns[d] = validate_pattern(make_seg7_digit(d), d)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        patterns[ch] = validate_pattern(make_led_letter(ch), ch)
    return patterns


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

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, GRID_N)
    ax.set_ylim(0, GRID_N)
    ax.set_aspect("equal")

    for r in range(GRID_N):
        for c in range(GRID_N):
            if not mask[r, c]:
                continue
            y0 = GRID_N - 1 - r
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

    for i in range(GRID_N + 1):
        ax.plot([0, GRID_N], [i, i], color="black", linewidth=1.0, zorder=10)
        ax.plot([i, i], [0, GRID_N], color="black", linewidth=1.0, zorder=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{name}  ({n_blocks} blocks)", fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_all(out_dir: Path, *, dpi: int = 120) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = collect_all_patterns()

    for name, pattern in sorted(patterns.items()):
        save_pattern_gt(pattern, out_dir / f"{name}.npy")
        render_pattern_png(pattern, name, out_dir / f"{name}.png", dpi=dpi)
        n_blocks = int(pattern_to_array(pattern).sum())
        print(f"  {name}: {n_blocks} blocks")

    keep = set(patterns.keys())
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

    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 20×20 / 1 cm 金属块布局 GT 图片"
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
    print(f"生成图案 → {args.out.resolve()}")
    patterns = generate_all(args.out, dpi=args.dpi)
    print(f"完成：{len(patterns)} 个图案（0–9, A–Z）")


if __name__ == "__main__":
    main()
