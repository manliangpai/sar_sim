#!/usr/bin/env python3
"""
运行示例（在仓库根目录执行）:

  python sar_sim/process_alpha_beta_r.py
  python sar_sim/process_alpha_beta_r.py --gpu
  python sar_sim/process_alpha_beta_r.py --gpu --input sar_sim/output/raw_radar_data/ti1843_2t4r_concave_room.npz --output sar_sim/output/processed_radar_data/ti1843_2t4r_concave_room_alpha_beta_r.npz

合成孔径雷达数据处理：距离 FFT + (alpha, beta, R) 网格双基地后向投影。

网格定义（相对原点 (0,0,0)，主瓣 +Z）：
  alpha — 原点→目标 与 +Z 夹角，[0°, 45°]，64 bins
  beta  — 目标在 XOY 投影与 +X 夹角，[0°, 360°)，512 bins
  R     — 目标到原点的欧式距离；256 bins，与距离 FFT 各 range bin 的
          等效单程斜距轴一致（process_alpha_beta_z.range_bins_to_slant_range_m）

alpha=0 时目标在 +Z 轴上，与 beta 无关（各 beta 列相同）。

输出 complex64 立方体 shape (64, 512, 256) 至
sar_sim/output/processed_radar_data/

GPU：``python sar_sim/process_alpha_beta_r.py --gpu``
"""

from __future__ import annotations

import argparse
import sys
import time
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
)
from sar_sim.process_alpha_beta_z import (
    _GpuBpState,
    _backproject_pixels,
    _backproject_pixels_gpu,
    _try_create_gpu_state,
    channel_tx_rx_positions_all_stops,
    load_sar_cube,
    range_bins_to_slant_range_m,
    range_fft_cube,
)
from sar_sim.simulate import (
    DEFAULT_OUTPUT_NAME,
    PROCESSED_RADAR_DIR,
    RAW_RADAR_DIR,
    ensure_output_dirs,
)

# --- 成像网格 (alpha, beta, R) ---
N_ALPHA = 64
N_BETA = 512
ALPHA_MIN_DEG = 0.0
ALPHA_MAX_DEG = 45.0
# R 维 bin 数 = 距离 FFT 快时间 bin 数（默认 256，与 RadarConfig.n_adc_samples 一致）

DEFAULT_INPUT = RAW_RADAR_DIR / DEFAULT_OUTPUT_NAME
DEFAULT_OUTPUT = PROCESSED_RADAR_DIR / "ti1843_2t4r_alpha_beta_r.npz"


def adc_effective_bandwidth_hz(radar: RadarConfig | None = None) -> float:
    """
    ADC 采样点在 ramp 上覆盖的瞬时带宽 (Hz)。

    B_adc = slope × T_adc，T_adc = n_adc_samples / adc_rate_hz。
    不用整段 ramp_time 的 chirp 总带宽。
    """
    radar = radar or RadarConfig()
    t_adc_s = radar.n_adc_samples / radar.adc_rate_hz
    return radar.slope_hz_per_s * t_adc_s


def range_bin_spacing_m(
    radar: RadarConfig | None = None,
    n_bins: int | None = None,
) -> float:
    """与 process_alpha_beta_z.range_bins_to_slant_range_m 相邻 bin 间距一致 (m)。"""
    radar = radar or RadarConfig()
    n_bins = n_bins or radar.n_adc_samples
    axis = range_bins_to_slant_range_m(n_bins, radar)
    return float(axis[1] - axis[0])


def range_resolution_m(radar: RadarConfig | None = None) -> float:
    """距离分辨率 ΔR = c / (2·B_adc)，B_adc 为 ADC 采样覆盖带宽。"""
    radar = radar or RadarConfig()
    bw = adc_effective_bandwidth_hz(radar)
    if bw <= 0.0:
        raise ValueError(f"无效 ADC 有效带宽 {bw} Hz")
    return C_LIGHT / (2.0 * bw)


def grid_axes_alpha_beta_r(
    radar: RadarConfig | None = None,
    n_range_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    返回 (alpha_deg, beta_deg, r_m, dr_m)。

    r_m 与距离 FFT 的 range 轴相同：range_bins_to_slant_range_m(n_range_bins)。
    默认 n_range_bins = radar.n_adc_samples（256）。
    """
    radar = radar or RadarConfig()
    n_range_bins = n_range_bins or radar.n_adc_samples
    r_m = range_bins_to_slant_range_m(n_range_bins, radar)
    dr = range_bin_spacing_m(radar, n_range_bins)
    alpha_deg = np.linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, N_ALPHA, dtype=np.float64)
    beta_deg = np.linspace(0.0, 360.0, N_BETA, endpoint=False, dtype=np.float64)
    return alpha_deg, beta_deg, r_m, dr


def cartesian_from_alpha_beta_r(
    alpha_rad: float,
    beta_rad: np.ndarray,
    r_m: float | np.ndarray,
) -> np.ndarray:
    """
    球坐标 (alpha, beta, R) → 笛卡尔 (x, y, z)。

    alpha：与 +Z 夹角；beta：XOY 内相对 +X 方位；R：到原点距离。
    alpha=0 → (0, 0, R)。

    beta_rad : (...,) 与 r_m 可广播
    返回 (..., 3)
    """
    beta_rad = np.asarray(beta_rad, dtype=np.float64)
    r = np.broadcast_to(np.asarray(r_m, dtype=np.float64), beta_rad.shape)
    if alpha_rad == 0.0:
        zeros = np.zeros_like(r)
        return np.stack([zeros, zeros, r], axis=-1)

    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    cos_b = np.cos(beta_rad)
    sin_b = np.sin(beta_rad)
    x = r * sin_a * cos_b
    y = r * sin_a * sin_b
    z = r * cos_a
    return np.stack([x, y, z], axis=-1)


def backproject_alpha_beta_r(
    range_cube: np.ndarray,
    alpha_deg: np.ndarray | None = None,
    beta_deg: np.ndarray | None = None,
    r_m: np.ndarray | None = None,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    progress: bool = True,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在 (alpha, beta, R) 网格上双基地后向投影。

    按 R 的 n_R 个 bin 逐层处理；每层对全部 (alpha, beta) 成像。

    Returns
    -------
    image : (n_alpha, n_beta, n_r) complex64
    alpha_deg, beta_deg, r_m
    """
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()

    if range_cube.shape[0] != rotation.n_stops:
        raise ValueError(
            f"停点数 {range_cube.shape[0]} 与 n_stops={rotation.n_stops} 不一致"
        )

    n_range = range_cube.shape[2]
    range_axis_m = range_bins_to_slant_range_m(n_range, radar)
    dr = range_bin_spacing_m(radar, n_range)

    if alpha_deg is None or beta_deg is None or r_m is None:
        alpha_deg, beta_deg, r_m, _ = grid_axes_alpha_beta_r(radar, n_range_bins=n_range)

    n_r = len(r_m)
    if len(alpha_deg) != N_ALPHA or len(beta_deg) != N_BETA:
        raise ValueError(
            f"期望 alpha/beta ({N_ALPHA}, {N_BETA})，"
            f"得到 ({len(alpha_deg)}, {len(beta_deg)})"
        )
    if n_r != n_range:
        raise ValueError(
            f"R 轴长度 {n_r} 须与距离 FFT bin 数 {n_range} 一致"
        )
    if not np.allclose(r_m, range_axis_m, rtol=0.0, atol=1e-9):
        raise ValueError(
            "r_m 须与 range_bins_to_slant_range_m(range_cube) 一致"
        )

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

    image = np.zeros((N_ALPHA, N_BETA, n_r), dtype=np.complex64)
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)
    t_bp = time.perf_counter()

    for ir, r_val in enumerate(r_m):
        if progress:
            pct = 100.0 * (ir + 1) / n_r
            elapsed = time.perf_counter() - t_bp
            eta = (elapsed / (ir + 1)) * (n_r - ir - 1) if ir > 0 else 0.0
            print(
                f"  R layer {ir + 1}/{n_r}  R = {r_val:.4f} m  "
                f"({pct:.1f}%, elapsed {elapsed:.1f} s, ETA {eta:.1f} s)",
                flush=True,
            )

        for ia, a_rad in enumerate(alpha_rad):
            if a_rad == 0.0:
                pixel = np.array([0.0, 0.0, float(r_val)], dtype=np.float64)
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
                image[ia, :, ir] = val
                continue

            pixels = cartesian_from_alpha_beta_r(a_rad, beta_rad, r_val)
            if gpu is not None:
                image[ia, :, ir] = _backproject_pixels_gpu(gpu, pixels)
            else:
                image[ia, :, ir] = _backproject_pixels(
                    range_cube,
                    range_axis_m,
                    phase_scale,
                    tx_pos,
                    rx_pos,
                    pixels,
                )

    return image, alpha_deg, beta_deg, r_m


def save_processed_cube(
    image: np.ndarray,
    alpha_deg: np.ndarray,
    beta_deg: np.ndarray,
    r_m: np.ndarray,
    path: Path,
    range_resolution_m_val: float | None = None,
    radar: RadarConfig | None = None,
) -> Path:
    ensure_output_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = dict(
        image=image.astype(np.complex64, copy=False),
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        r_m=r_m,
        grid_shape=np.array(image.shape, dtype=np.int32),
        n_range_bins=np.int32(len(r_m)),
        r_min_m=np.float64(r_m[0]),
        r_max_m=np.float64(r_m[-1]),
    )
    if range_resolution_m_val is not None:
        radar = radar or RadarConfig()
        kwargs["range_resolution_m"] = np.float64(range_resolution_m_val)
        kwargs["adc_effective_bandwidth_hz"] = np.float64(
            adc_effective_bandwidth_hz(radar)
        )
    np.savez_compressed(path, **kwargs)
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

    n_range = range_cube.shape[2]
    dr = range_bin_spacing_m(radar, n_range)
    bw_adc = adc_effective_bandwidth_hz(radar)
    alpha_deg, beta_deg, r_m, _ = grid_axes_alpha_beta_r(radar, n_range_bins=n_range)
    n_r = len(r_m)

    print(
        f"backproject (alpha, beta, R) = {N_ALPHA} x {N_BETA} x {n_r}, "
        f"B_adc = {bw_adc/1e9:.3f} GHz, dr = {dr*1000:.3f} mm, "
        f"R axis = range FFT bins [{r_m[0]:.3f}, {r_m[-1]:.3f}] m, "
        f"processing {n_r} R layers ...",
        flush=True,
    )
    t0 = time.perf_counter()
    image, alpha_deg, beta_deg, r_m = backproject_alpha_beta_r(
        range_cube,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        r_m=r_m,
        radar=radar,
        array=array,
        rotation=rotation,
        use_gpu=use_gpu,
    )
    elapsed = time.perf_counter() - t0
    peak = float(np.abs(image).max())
    print(f"  done in {elapsed:.1f} s, |I| peak = {peak:.4g}")

    out_path = save_processed_cube(
        image,
        alpha_deg,
        beta_deg,
        r_m,
        output_npz,
        range_resolution_m_val=dr,
        radar=radar,
    )
    print(f"saved {out_path}")
    print(f"  image shape {image.shape}, dtype {image.dtype}")
    return image, alpha_deg, beta_deg, r_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAR (alpha, beta, R) backprojection → processed npz"
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
