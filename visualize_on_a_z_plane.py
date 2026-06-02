#!/usr/bin/env python3
"""
运行示例（在 sar_sim 仓库根目录执行）::

  python visualize_on_a_z_plane.py
  python visualize_on_a_z_plane.py --npz output/raw_radar_data_z0.5/circle.npz
  python visualize_on_a_z_plane.py --npz output/raw_radar_data_z0.5/circle.npz --z 0.5

SAR 立方体可视化：距离 FFT + 固定 z 平面双基地后向投影。

默认 npz 见 DEFAULT_NPZ；成像 z / x/y 范围 / 格距见下方「成像网格」常量（本脚本专用，不读 config/scene）。
阵列与转台参数来自 sar_sim.config（2TX×4RX）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from sar_sim.config import (
    ArrayConfig,
    C_LIGHT,
    RadarConfig,
    SarRotationConfig,
    channel_tx_rx_index,
    rx_positions_at_stop,
    tx_positions_at_stop,
)
from sar_sim.simulate_pics import load_sar_cube

_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_NPZ = _PACKAGE_ROOT / "output" / "raw_radar_data_z0.5" / "circle.npz"

# ---------------------------------------------------------------------------
# 成像网格（本脚本专用，不依赖 config/scene）
# ---------------------------------------------------------------------------
IMAGING_Z_M = 0.5
XY_MIN_M = -0.2
XY_MAX_M = 0.2
CELL_STEP_M = 0.005

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


def range_fft_cube(raw: np.ndarray) -> np.ndarray:
    """快时间 Hanning + FFT，保持 (n_stops, 8, n_adc)。"""
    n_fast = raw.shape[2]
    win = np.hanning(n_fast).astype(np.float64)
    return np.fft.fft(raw * win[np.newaxis, np.newaxis, :], n=n_fast, axis=2).astype(
        np.complex64
    )


def range_bins_to_slant_range_m(
    n_bins: int, radar: RadarConfig | None = None
) -> np.ndarray:
    """FFT bin → 等效单程斜距 R=L/2 (m)。"""
    radar = radar or RadarConfig()
    k = np.arange(n_bins, dtype=np.float64)
    f_beat_hz = k * radar.adc_rate_hz / n_bins
    return f_beat_hz * C_LIGHT / (2.0 * radar.slope_hz_per_s)


def imaging_grid_axes() -> tuple[np.ndarray, np.ndarray]:
    """
    成像格心坐标轴 (m)。

    由 XY_MIN_M、XY_MAX_M、CELL_STEP_M 确定范围与格距；
    格点数 n = (XY_MAX - XY_MIN) / CELL_STEP。
    """
    n = int(round((XY_MAX_M - XY_MIN_M) / CELL_STEP_M))
    axis = (XY_MIN_M + (np.arange(n, dtype=np.float64) + 0.5) * CELL_STEP_M).astype(
        np.float64
    )
    return axis, axis.copy()


def channel_tx_rx_positions_all_stops(
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_stops, n_channels, 3) 各停点、各数据通道的 TX / RX 坐标。"""
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()
    n = rotation.n_stops
    nch = array.n_channels
    tx_all = np.zeros((n, nch, 3), dtype=np.float64)
    rx_all = np.zeros((n, nch, 3), dtype=np.float64)
    for k in range(1, n + 1):
        tx_pos = tx_positions_at_stop(array, k, rotation)
        rx_pos = rx_positions_at_stop(array, k, rotation)
        for ch in range(nch):
            ti, ri = channel_tx_rx_index(ch, array)
            tx_all[k - 1, ch] = tx_pos[ti]
            rx_all[k - 1, ch] = rx_pos[ri]
    return tx_all, rx_all


def _interp_range_cube(
    range_cube: np.ndarray,
    range_axis_m: np.ndarray,
    r_query_m: np.ndarray,
) -> np.ndarray:
    """沿距离维线性插值；r_query 末两维为 (n_stop, n_ch)。"""
    orig_shape = r_query_m.shape
    n_stop, n_ch = range_cube.shape[0], range_cube.shape[1]
    if orig_shape[-2:] != (n_stop, n_ch):
        raise ValueError(f"r_query 末维须为 ({n_stop}, {n_ch})")

    n_bins = range_cube.shape[2]
    n_leading = int(np.prod(orig_shape[:-2]))
    r_q = r_query_m.reshape(n_leading, n_stop, n_ch)
    out = np.zeros((n_leading, n_stop, n_ch), dtype=np.complex64)

    r0 = range_axis_m[0]
    dr = range_axis_m[1] - range_axis_m[0]

    for s in range(n_stop):
        for c in range(n_ch):
            rq = r_q[:, s, c]
            idx_f = (rq - r0) / dr
            valid = (idx_f >= 0.0) & (idx_f <= float(n_bins - 1))
            i0 = np.floor(idx_f).astype(np.int32)
            i0 = np.clip(i0, 0, n_bins - 2)
            w = (idx_f - i0).astype(np.float64)
            prof = range_cube[s, c]
            samp = (1.0 - w) * prof[i0] + w * prof[i0 + 1]
            out[:, s, c] = np.where(valid, samp, 0.0j)

    return out.reshape(orig_shape)


def backproject_z_plane(
    range_cube: np.ndarray,
    z_m: float = IMAGING_Z_M,
    x_axis: np.ndarray | None = None,
    y_axis: np.ndarray | None = None,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 z=z_m 平面上对距离压缩数据做双基地后向投影。

    几何：L = |TX-像素| + |RX-像素|；距离轴取 R=L/2 插值；
    相位补偿 exp(-j·2π/λ·L)，与 simulate 中 FMCW 载波项一致。
    """
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()

    if range_cube.shape[0] != rotation.n_stops:
        raise ValueError(
            f"停点数 {range_cube.shape[0]} 与 SarRotationConfig.n_stops={rotation.n_stops} 不一致"
        )

    if x_axis is None or y_axis is None:
        x_axis, y_axis = imaging_grid_axes()

    range_axis_m = range_bins_to_slant_range_m(range_cube.shape[2], radar)
    wavelength_m = radar.wavelength_m
    phase_scale = -2.0 * np.pi / wavelength_m

    tx_pos, rx_pos = channel_tx_rx_positions_all_stops(array, rotation)

    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    pixels = np.stack(
        [x_grid, y_grid, np.full_like(x_grid, float(z_m))],
        axis=-1,
    )

    diff_tx = (
        pixels[:, :, np.newaxis, np.newaxis, :]
        - tx_pos[np.newaxis, np.newaxis, :, :, :]
    )
    diff_rx = (
        pixels[:, :, np.newaxis, np.newaxis, :]
        - rx_pos[np.newaxis, np.newaxis, :, :, :]
    )
    r_tx = np.linalg.norm(diff_tx, axis=-1)
    r_rx = np.linalg.norm(diff_rx, axis=-1)
    path_len = r_tx + r_rx
    r_slant = path_len * 0.5

    samples = _interp_range_cube(range_cube, range_axis_m, r_slant)
    image = np.sum(
        samples * np.exp(1j * phase_scale * path_len),
        axis=(2, 3),
    )

    return image.astype(np.complex64, copy=False), x_axis, y_axis


def plot_z_plane(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_m: float = IMAGING_Z_M,
    *,
    display_floor_ratio: float = 0.0,
    interpolation: str = "nearest",
) -> None:
    """
    z=const 平面强度图。

    display_floor_ratio : 0 = 线性色标 [0, 峰值]；>0 仅用于压低旁瓣显示。
    interpolation='nearest' : 不做像素插值，避免格点被插值抹糊。
    """
    mag = np.abs(image)
    peak = float(mag.max()) if mag.size else 1.0
    vmin = peak * display_floor_ratio if peak > 0 else 0.0
    vmax = peak if peak > 0 else 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    im = ax.imshow(
        mag,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"SAR 后向投影 — z = {z_m:.1f} m（2TX×4RX 双基地）")
    plt.colorbar(im, ax=ax, label="|I|")
    plt.tight_layout()


def run_z_plane_view(
    npz_path: Path = DEFAULT_NPZ,
    z_m: float = IMAGING_Z_M,
    *,
    display_floor_ratio: float = 0.0,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup_chinese_font()
    radar = RadarConfig()
    array = ArrayConfig()
    print(f"loading {npz_path} ...")
    raw, start_angle_rad = load_sar_cube(npz_path)
    rotation = SarRotationConfig(
        n_stops=raw.shape[0], start_angle_rad=start_angle_rad
    )
    print(
        f"  n_stops={rotation.n_stops}, start_angle_rad={start_angle_rad:.6g} "
        f"({np.rad2deg(start_angle_rad):.4g} deg)"
    )
    print("range FFT ...")
    range_cube = range_fft_cube(raw)

    x_axis, y_axis = imaging_grid_axes()
    n = len(x_axis)
    print(
        f"backproject z = {z_m} m, x/y [{XY_MIN_M}, {XY_MAX_M}] m, "
        f"step {CELL_STEP_M} m → {n}×{n}, "
        f"{rotation.n_stops} stops x {array.n_channels} ch ..."
    )
    t0 = time.perf_counter()
    image, x_axis, y_axis = backproject_z_plane(
        range_cube,
        z_m=z_m,
        x_axis=x_axis,
        y_axis=y_axis,
        radar=radar,
        array=array,
        rotation=rotation,
    )
    elapsed = time.perf_counter() - t0
    peak = float(np.abs(image).max())
    print(f"  done in {elapsed:.1f} s, |I| peak = {peak:.4g}")

    plot_z_plane(
        image, x_axis, y_axis, z_m=z_m, display_floor_ratio=display_floor_ratio
    )
    if show:
        plt.show()
    return image, x_axis, y_axis


def main() -> None:
    parser = argparse.ArgumentParser(description="SAR raw cube → z 平面后向投影")
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help=f"raw SAR npz 路径（默认 {DEFAULT_NPZ}）",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=IMAGING_Z_M,
        metavar="M",
        help=f"成像 z 平面 (m)，默认 {IMAGING_Z_M}",
    )
    args = parser.parse_args()
    run_z_plane_view(npz_path=args.npz, z_m=args.z)


if __name__ == "__main__":
    main()
