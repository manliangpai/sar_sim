#!/usr/bin/env python3
"""
运行示例（在仓库根目录执行）::

  python sar_sim/visualize_alpha_beta_z.py
  python sar_sim/visualize_alpha_beta_z.py --npz sar_sim/output/processed_radar_data/ti1843_2t4r_alpha_beta_z.npz
  python sar_sim/visualize_alpha_beta_z.py --npz sar_sim/output/processed_radar_data/ti1843_2t4r_concave_room_alpha_beta_z.npz --floor 0.01

交互查看 processed (alpha, beta, Z) 成像立方体：底栏 Z 滑块，主图为 X–Y 平面 |I|。

默认数据：
  sar_sim/output/processed_radar_data/ti1843_2t4r_alpha_beta_z.npz

(alpha, beta) 按 process_alpha_beta_z.py 同一公式映射到 (x, y)；显示范围随 Z 变化
（约 ±Z·tan(alpha_max)），再插值到规则笛卡尔网格。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.widgets import Slider
from scipy.interpolate import griddata

from sar_sim.simulate import PROCESSED_RADAR_DIR

DEFAULT_NPZ = PROCESSED_RADAR_DIR / "ti1843_2t4r_alpha_beta_z.npz"

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
)


def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体（标题、坐标轴）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

XY_DISPLAY_N = 401
XY_PAD_FRAC = 0.06  # 显示范围在几何/数据包络外扩比例
Z0_PLANE_MIN_SPAN_M = 0.5  # z=0 时仍显示原点附近小窗口


def load_processed_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 image (n_alpha, n_beta, n_z), alpha_deg, beta_deg, z_m。"""
    archive = np.load(path)
    image = np.asarray(archive["image"], dtype=np.complex64)
    alpha_deg = np.asarray(archive["alpha_deg"], dtype=np.float64)
    beta_deg = np.asarray(archive["beta_deg"], dtype=np.float64)
    z_m = np.asarray(archive["z_m"], dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image 期望 3 维，得到 {image.shape}")
    return image, alpha_deg, beta_deg, z_m


def alpha_beta_slice_to_xy(
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    z_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    固定 Z，将 (n_alpha, n_beta) 网格节点映射到 XOY 坐标 (x, y)。

    与 process_alpha_beta_z.cartesian_from_alpha_beta_z 一致：
      rho = Z * tan(alpha),  x = rho*cos(beta),  y = rho*sin(beta)
    """
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)
    a_grid, b_grid = np.meshgrid(alpha_rad, beta_rad, indexing="ij")
    z = float(z_val)
    rho = z * np.tan(a_grid)
    x = rho * np.cos(b_grid)
    y = rho * np.sin(b_grid)
    return x, y


def xy_display_bounds(
    z_val: float,
    alpha_deg: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    pad_frac: float = XY_PAD_FRAC,
) -> tuple[float, float]:
    """
    随 Z 变化的 X/Y 显示半宽 (m)。

    几何上最远点：rho_max = Z * tan(alpha_max)，与 process 网格一致。
    """
    if z_val <= 0.0:
        half = Z0_PLANE_MIN_SPAN_M / 2.0
        return -half, half

    alpha_max_rad = np.deg2rad(float(alpha_deg[-1]))
    r_geom = float(z_val) * np.tan(alpha_max_rad)
    r_data = float(max(np.abs(x).max(initial=0.0), np.abs(y).max(initial=0.0)))
    half = max(r_geom, r_data) * (1.0 + pad_frac)
    return -half, half


def interpolate_to_cartesian_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int = XY_DISPLAY_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """不规则 (x,y) 采样 → 规则网格，用于 imshow。"""
    xi = np.linspace(x_min, x_max, n, dtype=np.float64)
    yi = np.linspace(y_min, y_max, n, dtype=np.float64)
    xi_grid, yi_grid = np.meshgrid(xi, yi, indexing="xy")
    pts = np.column_stack([x.ravel(), y.ravel()])
    v = values.ravel()
    zi = griddata(pts, v, (xi_grid, yi_grid), method="linear", fill_value=0.0)
    return xi, yi, zi


def slice_xy_image(
    image: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    z_m: np.ndarray,
    iz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """第 iz 层：返回 (xi, yi, |I| 网格), z_val, peak。"""
    iz = int(np.clip(iz, 0, len(z_m) - 1))
    z_val = float(z_m[iz])
    mag = np.abs(image[:, :, iz])
    x, y = alpha_beta_slice_to_xy(alpha_deg, beta_deg, z_val)
    x_lo, x_hi = xy_display_bounds(z_val, alpha_deg, x, y)
    xi, yi, zi = interpolate_to_cartesian_grid(
        x, y, mag, x_min=x_lo, x_max=x_hi, y_min=x_lo, y_max=x_hi
    )
    peak = float(zi.max()) if zi.size else 1.0
    return xi, yi, zi, z_val, peak, x_lo, x_hi


def run_interactive_view(
    npz_path: Path = DEFAULT_NPZ,
    *,
    display_floor_ratio: float = 0.0,
) -> None:
    setup_chinese_font()
    image, alpha_deg, beta_deg, z_m = load_processed_cube(npz_path)
    n_z = len(z_m)

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.14)
    iz0 = int(np.argmin(np.abs(z_m - 2.0))) if z_m.size else 0

    xi, yi, zi, z_val, peak, x_lo, x_hi = slice_xy_image(
        image, alpha_deg, beta_deg, z_m, iz0
    )
    vmin = peak * display_floor_ratio
    vmax = peak if peak > 0 else 1.0
    extent = [xi[0], xi[-1], yi[0], yi[-1]]
    im = ax.imshow(
        zi,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(x_lo, x_hi)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = ax.set_title(
        f"处理后 SAR 强度 — z = {z_val:.3f} m  (bin {iz0 + 1}/{n_z}), "
        f"|x|,|y| ≤ {x_hi:.2f} m"
    )
    cbar = plt.colorbar(im, ax=ax, label="|I|")

    ax_z = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(
        ax_z,
        "Z (m)",
        float(z_m[0]),
        float(z_m[-1]),
        valinit=z_val,
        valstep=z_m,
    )

    def _iz_from_slider(z: float) -> int:
        return int(np.argmin(np.abs(z_m - z)))

    def on_z_change(z: float) -> None:
        iz = _iz_from_slider(z)
        xi, yi, zi, zv, pk, x_lo, x_hi = slice_xy_image(
            image, alpha_deg, beta_deg, z_m, iz
        )
        im.set_data(zi)
        im.set_clim(pk * display_floor_ratio, pk if pk > 0 else 1.0)
        im.set_extent([xi[0], xi[-1], yi[0], yi[-1]])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(x_lo, x_hi)
        half = x_hi
        title.set_text(
            f"处理后 SAR 强度 — z = {zv:.3f} m  (bin {iz + 1}/{n_z}), "
            f"|x|,|y| ≤ {half:.2f} m"
        )
        fig.canvas.draw_idle()

    slider.on_changed(on_z_change)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="交互查看 (alpha,beta,Z) 处理结果 → X–Y 平面 + Z 滑块"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help="processed npz 路径",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="色标下限 = 峰值 × floor（0 表示 0~峰值线性）",
    )
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(
            f"未找到 {args.npz}，请先运行: python sar_sim/process_alpha_beta_z.py"
        )
    run_interactive_view(args.npz, display_floor_ratio=args.floor)


if __name__ == "__main__":
    main()
