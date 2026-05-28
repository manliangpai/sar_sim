#!/usr/bin/env python3
"""
停点 1 的 8 通道快照：距离 FFT + 远场方位-距离图（CBF）。

读取 sar_sim 仿真 npz，形状 (n_stops, 8, 256)；仅用第 1 个停点（小孔径，远场近似）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许: python sar_sim/test_code/temp_plot.py（项目根需在 sys.path）
if __name__ == "__main__" and __package__ is None:
    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

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

_SAR_SIM_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (
    _SAR_SIM_ROOT
    / "output"
    / "raw_radar_data"
    / "ti1843_2t4r_two_corner_reflectors.npz"
)

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)


def setup_chinese_font() -> None:
    """Windows 上优先使用系统已安装的中文字体，避免标题/坐标轴乱码。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def load_stop1_frame(path: Path = NPZ_PATH) -> np.ndarray:
    """(8, 256) complex，停点 1。"""
    data = np.load(path)["data"]
    rot = SarRotationConfig()
    expected = (rot.n_stops, 8, 256)
    if data.shape != expected:
        raise ValueError(f"期望 {expected}，得到 {data.shape}")
    return np.asarray(data[0], dtype=np.complex64)


def range_fft(raw: np.ndarray) -> np.ndarray:
    """Hanning + 快时间 FFT，(n_ch, n_adc)。"""
    n_samp = raw.shape[1]
    win = np.hanning(n_samp).astype(np.float64)
    return np.fft.fft(raw * win, n=n_samp, axis=1)


def range_bins_to_slant_range_m(n_bins: int, radar: RadarConfig | None = None) -> np.ndarray:
    """
    FFT bin → 等效单程斜距 (m)。

    双基地总路程 L≈2R 时，与单基地 FMCW 公式 R=c·f_b/(2·slope) 一致。
    """
    radar = radar or RadarConfig()
    k = np.arange(n_bins, dtype=np.float64)
    f_beat_hz = k * radar.adc_rate_hz / n_bins
    return f_beat_hz * C_LIGHT / (2.0 * radar.slope_hz_per_s)


def channel_bistatic_x_m(array: ArrayConfig | None = None, stop_index: int = 1) -> np.ndarray:
    """各通道远场相位参考：x_tx + x_rx（停点 stop_index）。"""
    array = array or ArrayConfig()
    tx = tx_positions_at_stop(array, stop_index)
    rx = rx_positions_at_stop(array, stop_index)
    xs = np.empty(array.n_channels, dtype=np.float64)
    for ch in range(array.n_channels):
        ti, ri = channel_tx_rx_index(ch, array)
        xs[ch] = tx[ti, 0] + rx[ri, 0]
    return xs


def azimuth_beamform_bistatic(
    range_data: np.ndarray,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    n_az_bins: int = 128,
    az_min_deg: float = -50.0,
    az_max_deg: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    远场 CBF：双基地路程差 ΔL = (x_tx+x_rx - x_ref)·sin(θ)，补偿后相干叠加。

    range_data : (8, n_range)
    返回 (n_az, n_range) 与方位角 (deg)。
    """
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    x_ch = channel_bistatic_x_m(array, stop_index=1)
    x_ref = x_ch[0]
    dx = x_ch - x_ref

    az_deg = np.linspace(az_min_deg, az_max_deg, n_az_bins)
    az_rad = np.deg2rad(az_deg)
    # 与仿真一致：总路程 L 的相位 ∝ 2π/λ · L；远场 ΔL ≈ dx·sin(θ)
    delta_l = dx[:, np.newaxis] * np.sin(az_rad)[np.newaxis, :]
    phase = np.exp(-1j * 2.0 * np.pi / radar.wavelength_m * delta_l)
    az_spec = (phase.conj().T @ range_data).astype(np.complex64)
    return az_spec, az_deg


def plot_range_fft_channels(
    range_data: np.ndarray,
    range_axis_m: np.ndarray,
    channel_labels: list[str],
) -> None:
    """一张图 8 个子图：各通道距离 FFT 幅度 vs 斜距。"""
    mag = np.abs(range_data)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle("停点 1 — 距离 FFT（Hanning，线性幅度）")

    for ch in range(8):
        ax = axes[ch // 4, ch % 4]
        ax.plot(range_axis_m, mag[ch], linewidth=0.9)
        ax.set_title(channel_labels[ch], fontsize=10)
        ax.grid(True, alpha=0.3)
        if ch // 4 == 1:
            ax.set_xlabel("斜距 (m)")
        if ch % 4 == 0:
            ax.set_ylabel("信号强度")

    plt.tight_layout()


def plot_azimuth_range(
    az_spec: np.ndarray,
    az_deg: np.ndarray,
    range_axis_m: np.ndarray,
) -> None:
    """方位角–距离图。"""
    mag = np.abs(az_spec)
    vmax = np.percentile(mag, 99.5) if mag.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(9, 5))
    extent = [
        range_axis_m[0],
        range_axis_m[-1],
        az_deg[0],
        az_deg[-1],
    ]
    im = ax.imshow(
        mag,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel("斜距 (m)")
    ax.set_ylabel("方位角 (deg)")
    ax.set_title("停点 1 — 方位-距离（远场 CBF，双基地 x_tx+x_rx）")
    plt.colorbar(im, ax=ax, label="幅度")
    plt.tight_layout()


def channel_labels(array: ArrayConfig | None = None) -> list[str]:
    array = array or ArrayConfig()
    labels = []
    for ch in range(array.n_channels):
        ti, ri = channel_tx_rx_index(ch, array)
        labels.append(f"ch{ch} T{ti+1}R{ri+1}")
    return labels


def main() -> None:
    setup_chinese_font()
    radar = RadarConfig()
    array = ArrayConfig()
    raw = load_stop1_frame()
    print(f"loaded {NPZ_PATH.name}, stop-1 shape {raw.shape}")

    rng = range_fft(raw)
    range_m = range_bins_to_slant_range_m(rng.shape[1], radar)
    labels = channel_labels(array)

    az_spec, az_deg = azimuth_beamform_bistatic(rng, radar=radar, array=array)
    print(f"range FFT peak: {np.abs(rng).max():.4g}")
    print(f"az-range peak:  {np.abs(az_spec).max():.4g}")

    plot_range_fft_channels(rng, range_m, labels)
    plot_azimuth_range(az_spec, az_deg, range_m)
    plt.show()


if __name__ == "__main__":
    main()
