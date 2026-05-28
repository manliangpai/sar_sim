#!/usr/bin/env python3
"""
运行示例（在仓库根目录执行）::

  python sar_sim/process_alpha_beta_z.py
  python sar_sim/process_alpha_beta_z.py --gpu
  python sar_sim/process_alpha_beta_z.py --gpu --input sar_sim/output/raw_radar_data/ti1843_2t4r_concave_room.npz --output sar_sim/output/processed_radar_data/ti1843_2t4r_concave_room_alpha_beta_z.npz

合成孔径雷达数据处理：距离 FFT + (alpha, beta, Z) 网格双基地后向投影。

网格定义（相对原点 (0,0,0)，主瓣 +Z）：
  alpha — 原点→目标 与 +Z 夹角，[0°, 45°]，64 bins
  beta  — 目标在 XOY 投影与 +X 夹角，[0°, 360°)，512 bins
  Z     — 目标 z 坐标，[0, 7.4] m，180 bins

alpha=0 时目标在 +Z 轴上，与 beta 无关（各 beta 列相同）。

输出 complex64 立方体 shape (64, 512, 180) 至
sar_sim/output/processed_radar_data/

GPU：安装 PyTorch CUDA 后使用 ``python sar_sim/process_alpha_beta_z.py --gpu``（与 CPU 算法相同，仅加速）。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

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
from sar_sim.simulate import (
    DEFAULT_OUTPUT_NAME,
    PROCESSED_RADAR_DIR,
    RAW_RADAR_DIR,
    ensure_output_dirs,
)

# --- 成像网格 (alpha, beta, Z) ---
N_ALPHA = 64
N_BETA = 512
N_Z = 180
ALPHA_MIN_DEG = 0.0
ALPHA_MAX_DEG = 45.0
Z_MIN_M = 0.0
Z_MAX_M = 7.4

DEFAULT_INPUT = RAW_RADAR_DIR / DEFAULT_OUTPUT_NAME
DEFAULT_OUTPUT = PROCESSED_RADAR_DIR / "ti1843_2t4r_alpha_beta_z.npz"


def load_sar_cube(path: Path) -> tuple[np.ndarray, float]:
    """
    读取原始 SAR npz。

    返回 ((n_stops, 8, 256) complex64, start_angle_rad)。
    旧文件无 start_angle_rad 字段时视为 0。
    """
    archive = np.load(path)
    data = archive["data"]
    if data.ndim != 3 or data.shape[1] != 8 or data.shape[2] != 256:
        raise ValueError(f"期望 (n_stops, 8, 256)，得到 {data.shape}")
    start_angle_rad = (
        float(archive["start_angle_rad"])
        if "start_angle_rad" in archive.files
        else 0.0
    )
    return np.asarray(data, dtype=np.complex64), start_angle_rad


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


@dataclass
class _GpuBpState:
    """常驻 GPU 的 SAR 立方体与天线坐标（后向投影复用）。"""

    range_cube: Any
    range_axis_m: Any
    tx_pos: Any
    rx_pos: Any
    phase_scale: float


def _try_create_gpu_state(
    range_cube: np.ndarray,
    range_axis_m: np.ndarray,
    phase_scale: float,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
) -> _GpuBpState | None:
    """CUDA 可用时上传数据；否则返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    except OSError as exc:
        print(
            f"PyTorch 加载失败（{exc}），回退 CPU。"
            " 可重装: pip install torch --index-url https://download.pytorch.org/whl/cu124",
            flush=True,
        )
        return None
    if not torch.cuda.is_available():
        return None
    dev = torch.device("cuda")
    return _GpuBpState(
        range_cube=torch.as_tensor(range_cube, device=dev),
        range_axis_m=torch.as_tensor(range_axis_m, dtype=torch.float64, device=dev),
        tx_pos=torch.as_tensor(tx_pos, dtype=torch.float32, device=dev),
        rx_pos=torch.as_tensor(rx_pos, dtype=torch.float32, device=dev),
        phase_scale=phase_scale,
    )


def _interp_range_cube_gpu(
    range_cube: Any,
    range_axis_m: Any,
    r_query_m: Any,
) -> Any:
    """GPU 向量化距离插值；r_query 已在 CUDA 上。"""
    import torch

    orig_shape = r_query_m.shape
    n_stop, n_ch = range_cube.shape[0], range_cube.shape[1]
    n_bins = range_cube.shape[2]
    r_q = r_query_m.reshape(-1, n_stop, n_ch)
    flat = range_cube.reshape(n_stop * n_ch, n_bins)
    dev = flat.device
    sc = torch.arange(n_stop, device=dev)[:, None] * n_ch + torch.arange(
        n_ch, device=dev
    )[None, :]

    r0 = range_axis_m[0]
    dr = range_axis_m[1] - range_axis_m[0]
    idx_f = (r_q - r0) / dr
    valid = (idx_f >= 0.0) & (idx_f <= float(n_bins - 1))
    i0 = torch.floor(idx_f).long().clamp(0, n_bins - 2)
    w = (idx_f - i0.to(idx_f.dtype)).to(flat.dtype)
    sc_bc = sc.reshape(1, n_stop, n_ch)
    left = flat[sc_bc, i0]
    right = flat[sc_bc, i0 + 1]
    samp = (1.0 - w) * left + w * right
    return torch.where(valid, samp, torch.zeros_like(samp)).reshape(orig_shape)


def _backproject_pixels_gpu(
    gpu: _GpuBpState,
    pixels: np.ndarray,
) -> np.ndarray:
    """与 CPU 版相同几何；pixels (n_pix, 3) → numpy complex64 (n_pix,)。"""
    import torch

    pix = torch.as_tensor(pixels, dtype=torch.float32, device=gpu.range_cube.device)
    diff_tx = pix[:, None, None, :] - gpu.tx_pos[None, :, :, :]
    diff_rx = pix[:, None, None, :] - gpu.rx_pos[None, :, :, :]
    path_len = torch.linalg.norm(diff_tx, dim=-1) + torch.linalg.norm(diff_rx, dim=-1)
    r_slant = path_len * 0.5
    samples = _interp_range_cube_gpu(gpu.range_cube, gpu.range_axis_m, r_slant)
    ph = torch.exp(1j * gpu.phase_scale * path_len)
    out = torch.sum(samples * ph, dim=(1, 2))
    return out.cpu().numpy().astype(np.complex64, copy=False)


def grid_axes_alpha_beta_z() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (alpha_deg, beta_deg, z_m)。"""
    alpha_deg = np.linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, N_ALPHA, dtype=np.float64)
    beta_deg = np.linspace(0.0, 360.0, N_BETA, endpoint=False, dtype=np.float64)
    z_m = np.linspace(Z_MIN_M, Z_MAX_M, N_Z, dtype=np.float64)
    return alpha_deg, beta_deg, z_m


def cartesian_from_alpha_beta_z(
    alpha_rad: float,
    beta_rad: np.ndarray,
    z_m: np.ndarray,
) -> np.ndarray:
    """
    (alpha, beta, Z) → (x, y, z)。

    beta_rad : (...,) 与 z_m 可广播（逐 Z 层时 beta 为 (512,) 而 z 为标量）
    z_m      : 标量或数组
    返回 (..., 3)
    """
    beta_rad = np.asarray(beta_rad, dtype=np.float64)
    z = np.broadcast_to(np.asarray(z_m, dtype=np.float64), beta_rad.shape)
    if alpha_rad == 0.0:
        zeros = np.zeros_like(z)
        return np.stack([zeros, zeros, z], axis=-1)

    tan_a = np.tan(alpha_rad)
    cos_b = np.cos(beta_rad)
    sin_b = np.sin(beta_rad)
    rho = z * tan_a
    x = rho * cos_b
    y = rho * sin_b
    return np.stack([x, y, z], axis=-1)


def _backproject_pixels(
    range_cube: np.ndarray,
    range_axis_m: np.ndarray,
    phase_scale: float,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    """pixels (n_pix, 3) → 复振幅 (n_pix,)。"""
    diff_tx = pixels[:, np.newaxis, np.newaxis, :] - tx_pos[np.newaxis, :, :, :]
    diff_rx = pixels[:, np.newaxis, np.newaxis, :] - rx_pos[np.newaxis, :, :, :]
    path_len = np.linalg.norm(diff_tx, axis=-1) + np.linalg.norm(diff_rx, axis=-1)
    r_slant = path_len * 0.5
    samples = _interp_range_cube(range_cube, range_axis_m, r_slant)
    return np.sum(
        samples * np.exp(1j * phase_scale * path_len),
        axis=(1, 2),
    )


def backproject_alpha_beta_z(
    range_cube: np.ndarray,
    alpha_deg: np.ndarray | None = None,
    beta_deg: np.ndarray | None = None,
    z_m: np.ndarray | None = None,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    progress: bool = True,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在 (alpha, beta, Z) 网格上双基地后向投影。

    按 Z 的 180 个 bin 逐层处理；每层对全部 (alpha, beta) 成像。

    Returns
    -------
    image : (n_alpha, n_beta, n_z) complex64
    alpha_deg, beta_deg, z_m
    """
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()

    if range_cube.shape[0] != rotation.n_stops:
        raise ValueError(
            f"停点数 {range_cube.shape[0]} 与 n_stops={rotation.n_stops} 不一致"
        )

    if alpha_deg is None or beta_deg is None or z_m is None:
        alpha_deg, beta_deg, z_m = grid_axes_alpha_beta_z()

    if len(alpha_deg) != N_ALPHA or len(beta_deg) != N_BETA or len(z_m) != N_Z:
        raise ValueError(
            f"期望网格 ({N_ALPHA}, {N_BETA}, {N_Z})，"
            f"得到 ({len(alpha_deg)}, {len(beta_deg)}, {len(z_m)})"
        )

    range_axis_m = range_bins_to_slant_range_m(range_cube.shape[2], radar)
    wavelength_m = radar.wavelength_m
    phase_scale = -2.0 * np.pi / wavelength_m
    tx_pos, rx_pos = channel_tx_rx_positions_all_stops(array, rotation)

    gpu: _GpuBpState | None = None
    if use_gpu:
        gpu = _try_create_gpu_state(
            range_cube, range_axis_m, phase_scale, tx_pos, rx_pos
        )
        if gpu is None:
            print("GPU 不可用，回退 CPU。", flush=True)
        else:
            import torch

            print(f"后向投影使用 GPU: {torch.cuda.get_device_name(0)}", flush=True)

    image = np.zeros((N_ALPHA, N_BETA, N_Z), dtype=np.complex64)
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)
    t_bp = time.perf_counter()

    for iz, z_val in enumerate(z_m):
        if progress:
            pct = 100.0 * (iz + 1) / N_Z
            elapsed = time.perf_counter() - t_bp
            eta = (elapsed / (iz + 1)) * (N_Z - iz - 1) if iz > 0 else 0.0
            print(
                f"  Z layer {iz + 1}/{N_Z}  z = {z_val:.4f} m  "
                f"({pct:.1f}%, elapsed {elapsed:.1f} s, ETA {eta:.1f} s)",
                flush=True,
            )

        for ia, a_rad in enumerate(alpha_rad):
            if a_rad == 0.0:
                pixel = np.array([0.0, 0.0, float(z_val)], dtype=np.float64)
                if gpu is not None:
                    val = _backproject_pixels_gpu(gpu, pixel[np.newaxis, :])[0]
                else:
                    val = _backproject_pixels(
                        range_cube,
                        range_axis_m,
                        phase_scale,
                        tx_pos,
                        rx_pos,
                        pixel[np.newaxis, :],
                    )[0]
                image[ia, :, iz] = val
                continue

            pixels = cartesian_from_alpha_beta_z(a_rad, beta_rad, z_val)
            if gpu is not None:
                image[ia, :, iz] = _backproject_pixels_gpu(gpu, pixels)
            else:
                image[ia, :, iz] = _backproject_pixels(
                    range_cube,
                    range_axis_m,
                    phase_scale,
                    tx_pos,
                    rx_pos,
                    pixels,
                )

    return image, alpha_deg, beta_deg, z_m


def save_processed_cube(
    image: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    z_m: np.ndarray,
    path: Path,
) -> Path:
    ensure_output_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=image.astype(np.complex64, copy=False),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        z_m=z_m,
        grid_shape=np.array(image.shape, dtype=np.int32),
    )
    return path


def run_process(
    input_npz: Path = DEFAULT_INPUT,
    output_npz: Path = DEFAULT_OUTPUT,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radar = RadarConfig()
    array = ArrayConfig()
    print(f"loading {input_npz}", flush=True)
    raw, start_angle_rad = load_sar_cube(input_npz)
    rotation = SarRotationConfig(
        n_stops=raw.shape[0], start_angle_rad=start_angle_rad
    )
    print(
        f"  n_stops={rotation.n_stops}, start_angle_rad={start_angle_rad:.6g} "
        f"({np.rad2deg(start_angle_rad):.4g} deg)",
        flush=True,
    )
    print("range FFT ...", flush=True)
    range_cube = range_fft_cube(raw)

    alpha_deg, beta_deg, z_m = grid_axes_alpha_beta_z()
    print(
        f"backproject (alpha, beta, Z) = {N_ALPHA} x {N_BETA} x {N_Z}, "
        f"processing {N_Z} Z layers ...",
        flush=True,
    )
    t0 = time.perf_counter()
    image, alpha_deg, beta_deg, z_m = backproject_alpha_beta_z(
        range_cube,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        z_m=z_m,
        radar=radar,
        array=array,
        rotation=rotation,
        use_gpu=use_gpu,
    )
    elapsed = time.perf_counter() - t0
    peak = float(np.abs(image).max())
    print(f"  done in {elapsed:.1f} s, |I| peak = {peak:.4g}")

    out_path = save_processed_cube(image, alpha_deg, beta_deg, z_m, output_npz)
    print(f"saved {out_path}")
    print(f"  image shape {image.shape}, dtype {image.dtype}")
    return image, alpha_deg, beta_deg, z_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAR (alpha, beta, Z) backprojection → processed npz"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="raw SAR cube npz (n_stops, 8, 256)，默认 600 停",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path under processed_radar_data",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="使用 NVIDIA GPU (PyTorch CUDA) 加速后向投影",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU（默认）",
    )
    args = parser.parse_args()
    use_gpu = args.gpu and not args.cpu
    run_process(args.input, args.output, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
