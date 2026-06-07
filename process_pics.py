#!/usr/bin/env python3
"""
金属板场景：GPU 距离 FFT + z 平面双基地后向投影。

读取 output/raw_radar_data/{PLY名}.npz，在固定 BP 网格上成像，
保存 (100, 100, 2) float32：通道 0=实部，通道 1=虚部。

BP 网格（本脚本专用）：
  z = 0.5 m，x/y ∈ [-0.2, +0.2] m，步长 0.004 m → 100×100。

运行（在 sar_sim 仓库根目录）::

  python process_pics.py
  python process_pics.py --raw output/raw_radar_data/metal_plate_20x20x5cm.npz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np

from sar_sim.config import (
    ArrayConfig,
    RadarConfig,
    SarRotationConfig,
    channel_tx_rx_index,
    rx_positions_at_stop,
    tx_positions_at_stop,
)
from sar_sim.config.scene import DEFAULT_PLY_PATH, raw_npz_stem, radar_output_dir
from sar_sim.simulate_pics import load_sar_cube
from sar_sim.visualize_on_a_z_plane import range_bins_to_slant_range_m

_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = radar_output_dir("raw_radar_data")
DEFAULT_OUTPUT_DIR = radar_output_dir("processed_radar_data")

# ---------------------------------------------------------------------------
# BP 成像网格（本脚本专用）
# ---------------------------------------------------------------------------
BP_Z_M = 0.5
BP_XY_MIN_M = -0.2
BP_XY_MAX_M = 0.2
BP_STEP_M = 0.004
BP_GRID_N = int(round((BP_XY_MAX_M - BP_XY_MIN_M) / BP_STEP_M))  # 100
BP_OUT_SHAPE = (BP_GRID_N, BP_GRID_N, 2)


def _require_cuda_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "process_pics 需要 PyTorch（CUDA）。请安装：pip install torch"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU，无法运行 process_pics。")
    return torch.device("cuda")


def _gpu_device_name() -> str:
    import torch

    _require_cuda_device()
    return torch.cuda.get_device_name(0)


def bp_imaging_axes() -> tuple[np.ndarray, np.ndarray]:
    """100×100 格心坐标轴 (m)。"""
    axis = (
        BP_XY_MIN_M + (np.arange(BP_GRID_N, dtype=np.float64) + 0.5) * BP_STEP_M
    ).astype(np.float64)
    return axis, axis.copy()


def complex_image_to_bp_data(image: np.ndarray) -> np.ndarray:
    """(100, 100) complex → (100, 100, 2) float32，[...,0]=Re，[...,1]=Im。"""
    if image.shape != (BP_GRID_N, BP_GRID_N):
        raise RuntimeError(
            f"BP 图像形状须 ({BP_GRID_N},{BP_GRID_N})，得到 {image.shape}"
        )
    return np.stack(
        [image.real.astype(np.float32), image.imag.astype(np.float32)],
        axis=-1,
    )


def _channel_tx_rx_positions_all_stops(
    array: ArrayConfig,
    rotation: SarRotationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_stops, n_channels, 3) 各停点、各通道 TX / RX。"""
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


def _range_fft_cube_gpu(raw: np.ndarray, device) -> np.ndarray:
    import torch

    t_raw = torch.as_tensor(raw, device=device)
    n_fast = t_raw.shape[2]
    win = torch.hann_window(n_fast, periodic=False, device=device, dtype=torch.float32)
    out = torch.fft.fft(t_raw * win, dim=2)
    return out.cpu().numpy().astype(np.complex64)


def _backproject_z_plane_gpu(
    range_cube: np.ndarray,
    *,
    z_m: float,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    radar: RadarConfig,
    array: ArrayConfig,
    rotation: SarRotationConfig,
    device,
) -> np.ndarray:
    import torch

    range_axis_m = range_bins_to_slant_range_m(range_cube.shape[2], radar)
    wavelength_m = radar.wavelength_m
    phase_scale = -2.0 * math.pi / wavelength_m

    tx_pos, rx_pos = _channel_tx_rx_positions_all_stops(array, rotation)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    pixels = np.stack(
        [x_grid, y_grid, np.full_like(x_grid, float(z_m))],
        axis=-1,
    )

    pix_t = torch.as_tensor(pixels, device=device, dtype=torch.float64)
    tx_t = torch.as_tensor(tx_pos, device=device, dtype=torch.float64)
    rx_t = torch.as_tensor(rx_pos, device=device, dtype=torch.float64)
    rc_t = torch.as_tensor(range_cube, device=device)

    diff_tx = pix_t[:, :, None, None, :] - tx_t[None, None, :, :, :]
    diff_rx = pix_t[:, :, None, None, :] - rx_t[None, None, :, :, :]
    r_tx = torch.linalg.norm(diff_tx, dim=-1)
    r_rx = torch.linalg.norm(diff_rx, dim=-1)
    path_len = r_tx + r_rx
    r_slant = path_len * 0.5

    height, width, n_stop, n_ch = (int(x) for x in r_slant.shape)
    p = height * width
    r_q = r_slant.reshape(p, n_stop, n_ch)
    r0 = float(range_axis_m[0])
    dr = float(range_axis_m[1] - range_axis_m[0])
    n_bins = range_cube.shape[2]

    samples = torch.zeros((p, n_stop, n_ch), dtype=torch.complex64, device=device)
    for s in range(n_stop):
        prof = rc_t[s]
        for c in range(n_ch):
            rq = r_q[:, s, c]
            idx_f = (rq - r0) / dr
            valid = (idx_f >= 0.0) & (idx_f <= float(n_bins - 1))
            i0 = torch.floor(idx_f).to(torch.int64).clamp(0, n_bins - 2)
            frac = (idx_f - i0.to(idx_f.dtype)).to(prof.dtype)
            s0 = prof[c, i0]
            s1 = prof[c, i0 + 1]
            samp = (1.0 - frac) * s0 + frac * s1
            samples[:, s, c] = torch.where(valid, samp, torch.zeros_like(samp))

    image = torch.sum(
        samples * torch.exp(1j * phase_scale * path_len.reshape(p, n_stop, n_ch)),
        dim=(1, 2),
    ).reshape(height, width)
    return image.cpu().numpy().astype(np.complex64)


def process_raw(
    raw_path: Path,
    *,
    output_dir: Path,
    z_m: float = BP_Z_M,
) -> tuple[np.ndarray, Path]:
    """读取 raw → GPU 距离 FFT → GPU z 平面 BP → (100,100,2)。"""
    in_path = raw_path
    if not in_path.is_file():
        raise FileNotFoundError(
            f"缺少 raw 数据：{in_path}\n"
            f"请先运行：python simulate_pics.py"
        )

    raw, start_angle_rad = load_sar_cube(in_path)
    rotation = SarRotationConfig(
        n_stops=raw.shape[0], start_angle_rad=start_angle_rad
    )
    x_axis, y_axis = bp_imaging_axes()
    radar = RadarConfig()
    array = ArrayConfig()

    device = _require_cuda_device()
    range_cube = _range_fft_cube_gpu(raw, device)
    image = _backproject_z_plane_gpu(
        range_cube,
        z_m=z_m,
        x_axis=x_axis,
        y_axis=y_axis,
        radar=radar,
        array=array,
        rotation=rotation,
        device=device,
    )

    data = complex_image_to_bp_data(image)
    out_path = output_dir / f"{in_path.stem}.npz"
    np.savez_compressed(out_path, data=data)
    return data, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="金属板 raw → GPU z 平面 BP (100×100×2 Re/Im)"
    )
    default_raw = DEFAULT_INPUT_DIR / f"{raw_npz_stem(DEFAULT_PLY_PATH)}.npz"
    parser.add_argument(
        "--raw",
        type=Path,
        default=default_raw,
        help=f"仿真 raw npz（默认 {default_raw.name}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"处理后输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=BP_Z_M,
        metavar="M",
        help=f"成像 z 平面 (m)，默认 {BP_Z_M}",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  raw   : {args.raw.resolve()}")
    print(f"  output: {args.output_dir.resolve()}")
    print(f"  gpu   : {_gpu_device_name()}")
    print(
        f"  BP    : {BP_GRID_N}×{BP_GRID_N}×2, "
        f"x/y [{BP_XY_MIN_M}, {BP_XY_MAX_M}] m, step {BP_STEP_M} m, z={args.z} m"
    )

    t0 = time.perf_counter()
    data, out_path = process_raw(
        args.raw,
        output_dir=args.output_dir,
        z_m=args.z,
    )
    elapsed = time.perf_counter() - t0
    mag = np.hypot(data[..., 0], data[..., 1])
    print(
        f"saved {out_path}  peak={float(mag.max()):.4g}  ({elapsed:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
