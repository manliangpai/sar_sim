#!/usr/bin/env python3
"""
运行示例（在仓库根目录执行）::

  python sar_sim/visualize_alpha_beta_r.py
  python sar_sim/visualize_alpha_beta_r.py --npz sar_sim/output/processed_radar_data/ti1843_2t4r_alpha_beta_r.npz
  python sar_sim/visualize_alpha_beta_r.py --npz sar_sim/output/processed_radar_data/ti1843_2t4r_concave_room_alpha_beta_r.npz --alpha 26

交互查看 processed (alpha, beta, R) 成像立方体：底栏 alpha 滑块，主图为 beta–R 极坐标 |I|。

默认数据：
  sar_sim/output/processed_radar_data/ti1843_2t4r_alpha_beta_r.npz

固定 alpha 切片：image[ia, :, :] 为 (n_beta, n_r)，极角 = beta，径向 = R。
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

from sar_sim.simulate import PROCESSED_RADAR_DIR

DEFAULT_NPZ = PROCESSED_RADAR_DIR / "ti1843_2t4r_alpha_beta_r.npz"

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


def load_processed_cube(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 image (n_alpha, n_beta, n_r), alpha_deg, beta_deg, r_m。"""
    archive = np.load(path)
    image = np.asarray(archive["image"], dtype=np.complex64)
    alpha_deg = np.asarray(archive["alpha_deg"], dtype=np.float64)
    beta_deg = np.asarray(archive["beta_deg"], dtype=np.float64)
    if "r_m" not in archive:
        raise KeyError(f"{path} 缺少 r_m，请使用 process_alpha_beta_r 生成的 npz")
    r_m = np.asarray(archive["r_m"], dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image 期望 3 维，得到 {image.shape}")
    n_a, n_b, n_r = image.shape
    if len(alpha_deg) != n_a or len(beta_deg) != n_b or len(r_m) != n_r:
        raise ValueError(
            f"轴长与 image 不一致: image {image.shape}, "
            f"alpha {len(alpha_deg)}, beta {len(beta_deg)}, r {len(r_m)}"
        )
    return image, alpha_deg, beta_deg, r_m


def slice_beta_r_polar(
    image: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    r_m: np.ndarray,
    ia: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    第 ia 个 alpha 切片 → 极坐标 pcolormesh 用 (beta_rad, r_m, |I|)。

    返回 beta_rad (n_beta,), r_m, Z (n_r, n_beta), alpha_val, peak。
    """
    ia = int(np.clip(ia, 0, len(alpha_deg) - 1))
    alpha_val = float(alpha_deg[ia])
    mag = np.abs(image[ia, :, :]).astype(np.float64)  # (n_beta, n_r)
    beta_rad = np.deg2rad(beta_deg)
    # pcolormesh(shading='auto'): Z 形状 (len(r), len(beta))
    z = mag.T
    peak = float(z.max()) if z.size else 1.0
    return beta_rad, r_m, z, alpha_val, peak


def run_interactive_view(
    npz_path: Path = DEFAULT_NPZ,
    *,
    display_floor_ratio: float = 0.0,
    initial_alpha_deg: float | None = None,
) -> None:
    setup_chinese_font()
    image, alpha_deg, beta_deg, r_m = load_processed_cube(npz_path)
    n_alpha = len(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)

    if initial_alpha_deg is None:
        ia0 = 0
    else:
        ia0 = int(np.argmin(np.abs(alpha_deg - initial_alpha_deg)))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    plt.subplots_adjust(bottom=0.14)

    _, _, z0, a0, peak = slice_beta_r_polar(
        image, alpha_deg, beta_deg, r_m, ia0
    )
    vmin = peak * display_floor_ratio
    vmax = peak if peak > 0 else 1.0
    pcm = ax.pcolormesh(
        beta_rad,
        r_m,
        z0,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_theta_zero_location("E")  # beta=0 → +X
    ax.set_theta_direction(1)  # 逆时针，与 beta 定义一致
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_xlabel("")
    ax.set_ylabel("R (m)", labelpad=28)
    title = ax.set_title(
        f"处理后 SAR 强度 — alpha = {a0:.2f}°  (bin {ia0 + 1}/{n_alpha}), "
        f"beta–R 极坐标"
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1, label="|I|")

    ax_alpha = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(
        ax_alpha,
        "alpha (°)",
        float(alpha_deg[0]),
        float(alpha_deg[-1]),
        valinit=a0,
        valstep=alpha_deg,
    )

    def _ia_from_slider(deg: float) -> int:
        return int(np.argmin(np.abs(alpha_deg - deg)))

    def on_alpha_change(deg: float) -> None:
        ia = _ia_from_slider(deg)
        _, _, z, av, pk = slice_beta_r_polar(
            image, alpha_deg, beta_deg, r_m, ia
        )
        pcm.set_array(z)
        pcm.set_clim(pk * display_floor_ratio, pk if pk > 0 else 1.0)
        title.set_text(
            f"处理后 SAR 强度 — alpha = {av:.2f}°  (bin {ia + 1}/{n_alpha}), "
            f"beta–R 极坐标"
        )
        fig.canvas.draw_idle()

    slider.on_changed(on_alpha_change)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="交互查看 (alpha,beta,R) 处理结果 → beta–R 极坐标 + alpha 滑块"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help="process_alpha_beta_r 输出的 npz 路径",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="色标下限 = 峰值 × floor（0 表示 0~峰值线性）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="初始 alpha (deg)，默认最小角",
    )
    args = parser.parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(
            f"未找到 {args.npz}，请先运行: python sar_sim/process_alpha_beta_r.py"
        )
    run_interactive_view(
        args.npz,
        display_floor_ratio=args.floor,
        initial_alpha_deg=args.alpha,
    )


if __name__ == "__main__":
    main()
