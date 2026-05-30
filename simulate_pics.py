#!/usr/bin/env python3
"""
pixel_pattern SAR FMCW 前向仿真与批量采集。

2TX×4RX：每个停点 TX1/TX2 各 1 chirp → (n_stops, 8, n_adc) complex64。

运行（在 sar_sim 仓库根目录）::

  python simulate_pics.py --gpu
  python simulate_pics.py --skip-existing
  python simulate_pics.py --pattern 0 A
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np

from sar_sim.config import (
    ArrayConfig,
    C_LIGHT,
    PointTarget,
    RadarConfig,
    SarRotationConfig,
    adc_time_axis,
    angle_deg_at_stop,
    monostatic_channel_gain,
    pixel_pattern_scene,
    rx_positions_at_stop,
    targets_to_xyz,
    tx_positions_at_stop,
)
from sar_sim.config.scene import (
    DEFAULT_CUBE_SUBDIV,
    DEFAULT_PATTERN_PICS_DIR,
    pixel_pattern_names,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_RADAR_ROOT = _PACKAGE_ROOT / "output" / "raw_radar_data_z0.6"
RAW_RADAR_DIR = RAW_RADAR_ROOT
DEFAULT_PICS_DIR = DEFAULT_PATTERN_PICS_DIR
DEFAULT_OUTPUT_DIR = RAW_RADAR_DIR


def ensure_output_dirs() -> None:
    RAW_RADAR_DIR.mkdir(parents=True, exist_ok=True)


def _gpu_status_label(use_gpu: bool) -> str:
    if not use_gpu:
        return "False"
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "unavailable"


def load_sar_cube(path: Path) -> tuple[np.ndarray, float]:
    """
    读取原始 SAR npz。

    返回 ((n_stops, 8, n_adc) complex64, start_angle_rad)。
    """
    archive = np.load(path)
    data = archive["data"]
    if data.ndim != 3 or data.shape[1] != 8:
        raise ValueError(f"期望 (n_stops, 8, n_adc)，得到 {data.shape}")
    start_angle_rad = (
        float(archive["start_angle_rad"])
        if "start_angle_rad" in archive.files
        else 0.0
    )
    return np.asarray(data, dtype=np.complex64), start_angle_rad


def _simulate_channels_for_one_tx(
    radar: RadarConfig,
    array: ArrayConfig,
    targets: Sequence[PointTarget],
    tx_pos: np.ndarray,
    rx_positions: np.ndarray,
    c: float = C_LIGHT,
) -> np.ndarray:
    """单 TX → (n_rx, n_adc) complex64。"""
    rx_positions = np.asarray(rx_positions, dtype=np.float64)
    if rx_positions.shape != (array.n_rx, 3):
        raise ValueError(f"rx_positions must be ({array.n_rx}, 3)")

    t = adc_time_axis(radar)
    if t[-1] > radar.ramp_time_s:
        raise ValueError(
            f"ADC 末点 {t[-1]*1e6:.3f} us 超出 ramp {radar.ramp_time_s*1e6:.3f} us"
        )

    targets_xyz = targets_to_xyz(list(targets))
    target_amp = np.array([tg.amplitude for tg in targets], dtype=np.float64)
    tx_pos = np.asarray(tx_pos, dtype=np.float64)
    pattern = radar.antenna_pattern
    r_tx = np.linalg.norm(targets_xyz - tx_pos, axis=1)

    data = np.zeros((array.n_rx, radar.n_adc_samples), dtype=np.complex64)

    for rx_j, ant in enumerate(rx_positions):
        r_rx = np.linalg.norm(targets_xyz - ant, axis=1)
        path_len = r_tx + r_rx
        delta_t = path_len / c

        b = (
            2.0 * np.pi * radar.f_start_hz * delta_t
            - np.pi * radar.slope_hz_per_s * delta_t * delta_t
        )
        c_coef = 2.0 * np.pi * radar.slope_hz_per_s * delta_t
        phase = b[:, np.newaxis] + c_coef[:, np.newaxis] * t[np.newaxis, :]

        if radar.use_antenna_pattern:
            gains = np.empty(targets_xyz.shape[0], dtype=np.float64)
            for i, tgt in enumerate(targets_xyz):
                g, _, _ = monostatic_channel_gain(tx_pos, ant, tgt, pattern)
                gains[i] = g
        else:
            gains = np.ones(targets_xyz.shape[0], dtype=np.float64)

        amp = target_amp * gains / (r_tx * r_rx)
        data[rx_j, :] = np.sum(amp[:, np.newaxis] * np.exp(1j * phase), axis=0).astype(
            np.complex64
        )

    return data


def _simulate_one_stop(
    radar: RadarConfig,
    array: ArrayConfig,
    targets: Sequence[PointTarget],
    stop_index: int,
    rotation: SarRotationConfig,
    c: float = C_LIGHT,
) -> np.ndarray:
    """一个停点：TX1 + TX2 → (8, n_adc)。"""
    tx_pos = tx_positions_at_stop(array, stop_index, rotation)
    rx_pos = rx_positions_at_stop(array, stop_index, rotation)
    frame = np.zeros((array.n_channels, radar.n_adc_samples), dtype=np.complex64)

    for tx_i in range(array.n_tx):
        block = _simulate_channels_for_one_tx(
            radar, array, targets, tx_pos[tx_i], rx_pos, c=c
        )
        frame[tx_i * array.n_rx : (tx_i + 1) * array.n_rx, :] = block

    return frame


@dataclass
class _GpuSimState:
    targets_xyz: object
    target_amp: object
    t_adc: object
    f_start_hz: float
    slope_hz_per_s: float
    c_light: float
    use_antenna_pattern: bool
    half_az_rad: float
    half_el_rad: float
    cutoff_multiplier: float
    device: object


def _try_create_gpu_sim_state(
    radar: RadarConfig,
    targets: Sequence[PointTarget],
    c: float = C_LIGHT,
) -> _GpuSimState | None:
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch 未安装（{exc}），回退 CPU。", flush=True)
        return None
    if not torch.cuda.is_available():
        return None

    targets_xyz = targets_to_xyz(list(targets))
    target_amp = np.array([tg.amplitude for tg in targets], dtype=np.float64)
    t_adc = adc_time_axis(radar)
    pattern = radar.antenna_pattern
    dev = torch.device("cuda")
    return _GpuSimState(
        targets_xyz=torch.as_tensor(targets_xyz, dtype=torch.float64, device=dev),
        target_amp=torch.as_tensor(target_amp, dtype=torch.float64, device=dev),
        t_adc=torch.as_tensor(t_adc, dtype=torch.float64, device=dev),
        f_start_hz=radar.f_start_hz,
        slope_hz_per_s=radar.slope_hz_per_s,
        c_light=c,
        use_antenna_pattern=radar.use_antenna_pattern,
        half_az_rad=math.radians(pattern.half_angle_az_deg),
        half_el_rad=math.radians(pattern.half_angle_el_deg),
        cutoff_multiplier=pattern.cutoff_multiplier,
        device=dev,
    )


def _axis_gain_torch(angle_rad, half_angle_rad: float, cutoff_multiplier: float):
    import torch

    if half_angle_rad <= 0:
        return torch.zeros_like(angle_rad)
    normalized = torch.abs(angle_rad) / half_angle_rad
    out = torch.cos(math.pi * normalized / 4.0) ** 2
    return torch.where(normalized >= cutoff_multiplier, torch.zeros_like(out), out)


def _direction_gain_torch(
    ant_xyz,
    targets_xyz,
    half_az_rad: float,
    half_el_rad: float,
    cutoff_multiplier: float,
):
    import torch

    diff = targets_xyz - ant_xyz
    dx, dy, dz = diff[:, 0], diff[:, 1], diff[:, 2]
    ground = torch.sqrt(dx * dx + dz * dz)
    az = torch.atan2(dx, dz)
    el = torch.atan2(dy, ground)
    return _axis_gain_torch(az, half_az_rad, cutoff_multiplier) * _axis_gain_torch(
        el, half_el_rad, cutoff_multiplier
    )


def _direction_gain_rx_torch(
    rx_xyz,
    targets_xyz,
    half_az_rad: float,
    half_el_rad: float,
    cutoff_multiplier: float,
):
    import torch

    diff = targets_xyz[:, None, :] - rx_xyz[None, :, :]
    dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]
    ground = torch.sqrt(dx * dx + dz * dz)
    az = torch.atan2(dx, dz)
    el = torch.atan2(dy, ground)
    return _axis_gain_torch(az, half_az_rad, cutoff_multiplier) * _axis_gain_torch(
        el, half_el_rad, cutoff_multiplier
    )


def _simulate_channels_for_one_tx_gpu(
    gpu: _GpuSimState,
    tx_pos: np.ndarray,
    rx_positions: np.ndarray,
) -> np.ndarray:
    import torch

    tx = torch.as_tensor(tx_pos, dtype=torch.float64, device=gpu.device)
    rx = torch.as_tensor(rx_positions, dtype=torch.float64, device=gpu.device)

    diff_tx = gpu.targets_xyz - tx
    r_tx = torch.linalg.norm(diff_tx, dim=1)
    diff_rx = gpu.targets_xyz[:, None, :] - rx[None, :, :]
    r_rx = torch.linalg.norm(diff_rx, dim=-1)
    path_len = r_tx[:, None] + r_rx
    delta_t = path_len / gpu.c_light

    b = (
        2.0 * math.pi * gpu.f_start_hz * delta_t
        - math.pi * gpu.slope_hz_per_s * delta_t * delta_t
    )
    c_coef = 2.0 * math.pi * gpu.slope_hz_per_s * delta_t
    phase = b.unsqueeze(-1) + c_coef.unsqueeze(-1) * gpu.t_adc

    if gpu.use_antenna_pattern:
        g_tx = _direction_gain_torch(
            tx,
            gpu.targets_xyz,
            gpu.half_az_rad,
            gpu.half_el_rad,
            gpu.cutoff_multiplier,
        )
        g_rx = _direction_gain_rx_torch(
            rx,
            gpu.targets_xyz,
            gpu.half_az_rad,
            gpu.half_el_rad,
            gpu.cutoff_multiplier,
        )
        gains = torch.sqrt(g_tx[:, None] * g_rx)
    else:
        gains = torch.ones_like(r_rx)

    amp = gpu.target_amp[:, None] * gains / (r_tx[:, None] * r_rx)
    data = torch.sum(amp.unsqueeze(-1) * torch.exp(1j * phase), dim=0)
    return data.cpu().numpy().astype(np.complex64)


def _simulate_one_stop_gpu(
    gpu: _GpuSimState,
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig,
) -> np.ndarray:
    tx_pos = tx_positions_at_stop(array, stop_index, rotation)
    rx_pos = rx_positions_at_stop(array, stop_index, rotation)
    frame = np.zeros((array.n_channels, gpu.t_adc.shape[0]), dtype=np.complex64)
    for tx_i in range(array.n_tx):
        block = _simulate_channels_for_one_tx_gpu(gpu, tx_pos[tx_i], rx_pos)
        frame[tx_i * array.n_rx : (tx_i + 1) * array.n_rx, :] = block
    return frame


def _simulate_sar_rotation_cube_gpu(
    radar: RadarConfig,
    array: ArrayConfig,
    rotation: SarRotationConfig,
    targets: Sequence[PointTarget],
    c: float = C_LIGHT,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    import torch

    gpu = _try_create_gpu_sim_state(radar, targets, c=c)
    if gpu is None:
        print("GPU 不可用，回退 CPU。", flush=True)
        return simulate_sar_rotation_cube(
            radar=radar,
            array=array,
            rotation=rotation,
            targets=targets,
            c=c,
            progress_callback=progress_callback,
            use_gpu=False,
        )

    cube = np.zeros(
        (rotation.n_stops, array.n_channels, radar.n_adc_samples),
        dtype=np.complex64,
    )
    for k in range(1, rotation.n_stops + 1):
        cube[k - 1] = _simulate_one_stop_gpu(gpu, array, k, rotation)
        if progress_callback is not None:
            progress_callback(k, rotation.n_stops)
    return cube


def simulate_sar_rotation_cube(
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    targets: Sequence[PointTarget] | None = None,
    c: float = C_LIGHT,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    use_gpu: bool = False,
) -> np.ndarray:
    """停采转台 SAR：返回 (n_stops, 8, n_adc_samples) complex64。"""
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()
    if targets is None:
        raise ValueError("targets 不能为 None")
    target_list = list(targets)

    if use_gpu:
        return _simulate_sar_rotation_cube_gpu(
            radar,
            array,
            rotation,
            target_list,
            c=c,
            progress_callback=progress_callback,
        )

    cube = np.zeros(
        (rotation.n_stops, array.n_channels, radar.n_adc_samples),
        dtype=np.complex64,
    )

    for k in range(1, rotation.n_stops + 1):
        cube[k - 1] = _simulate_one_stop(
            radar, array, target_list, k, rotation, c=c
        )
        if progress_callback is not None:
            progress_callback(k, rotation.n_stops)

    return cube


def save_raw_cube(
    data: np.ndarray,
    filename: str,
    output_dir: Path | None = None,
    *,
    start_angle_rad: float = 0.0,
) -> Path:
    out_dir = output_dir or RAW_RADAR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    np.savez_compressed(
        path,
        data=data.astype(np.complex64, copy=False),
        start_angle_rad=np.float64(start_angle_rad),
    )
    return path


def simulate_one(
    pattern_name: str,
    *,
    output_dir: Path,
    pics_dir: Path | None = None,
    cube_subdiv: int = DEFAULT_CUBE_SUBDIV,
    start_angle_rad: float = 0.0,
    use_gpu: bool = False,
) -> tuple[np.ndarray, Path, float]:
    targets = pixel_pattern_scene(
        pattern_name, pics_dir=pics_dir, cube_subdiv=cube_subdiv
    )
    radar = RadarConfig()
    array = ArrayConfig()
    rotation = SarRotationConfig(start_angle_rad=start_angle_rad)

    print(
        f"    scatterers: {len(targets)}, "
        f"cube_subdiv={cube_subdiv}, stops={rotation.n_stops}",
        flush=True,
    )

    def progress(k: int, n: int) -> None:
        if k == 1 or k == n or k % 100 == 0:
            print(
                f"    stop {k}/{n} ({angle_deg_at_stop(k, rotation):.1f} deg)",
                flush=True,
            )

    t0 = time.perf_counter()
    cube = simulate_sar_rotation_cube(
        radar=radar,
        array=array,
        rotation=rotation,
        targets=targets,
        progress_callback=progress,
        use_gpu=use_gpu,
    )
    path = save_raw_cube(
        cube,
        f"{pattern_name}.npz",
        output_dir=output_dir,
        start_angle_rad=start_angle_rad,
    )
    elapsed = time.perf_counter() - t0
    return cube, path, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量 pixel_pattern 仿真 → output/raw_radar_data_z0.6"
    )
    parser.add_argument(
        "--pattern",
        nargs="*",
        default=None,
        metavar="NAME",
        help="图案名（默认 output/pics 下全部）",
    )
    parser.add_argument(
        "--pics-dir",
        type=Path,
        default=DEFAULT_PICS_DIR,
        metavar="DIR",
        help=f"GT npy 目录（默认 {DEFAULT_PICS_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--cube-subdiv",
        type=int,
        default=DEFAULT_CUBE_SUBDIV,
        metavar="N",
        help=f"每 1 cm 块体素 N³，默认 {DEFAULT_CUBE_SUBDIV}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="已存在 {NAME}.npz 则跳过",
    )
    parser.add_argument(
        "--start-angle-rad",
        type=float,
        default=0.0,
        metavar="RAD",
        help="第 1 停转角 (rad)，默认 0",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="使用 NVIDIA GPU (PyTorch CUDA) 加速前向仿真",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU（覆盖 --gpu）",
    )
    args = parser.parse_args()

    if not (0.0 <= args.start_angle_rad < 2.0 * math.pi):
        raise SystemExit("start-angle-rad 须在 [0, 2π) 内")
    if args.cube_subdiv < 1:
        raise SystemExit("cube-subdiv 须 >= 1")

    pics_dir = args.pics_dir
    output_dir = args.output_dir

    names = args.pattern if args.pattern else pixel_pattern_names(pics_dir)
    if not names:
        raise SystemExit("未找到图案，请先运行 test_code/generate_pattern_pics.py")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"simulate {len(names)} patterns → {output_dir.resolve()}")
    print(f"  pics   : {pics_dir.resolve()}")
    use_gpu = args.gpu and not args.cpu
    print(
        f"  cube_subdiv={args.cube_subdiv}, gpu={_gpu_status_label(use_gpu)}"
    )

    n_done = 0
    n_skip = 0

    for i, name in enumerate(names, 1):
        out_path = output_dir / f"{name}.npz"
        if args.skip_existing and out_path.is_file():
            n_skip += 1
            print(f"[{i}/{len(names)}] {name}: skip (exists)", flush=True)
            continue

        print(f"[{i}/{len(names)}] {name} ...", flush=True)

        cube, path, elapsed = simulate_one(
            name,
            output_dir=output_dir,
            pics_dir=pics_dir,
            cube_subdiv=args.cube_subdiv,
            start_angle_rad=args.start_angle_rad,
            use_gpu=use_gpu,
        )
        n_done += 1
        print(f"    {name}: saved", flush=True)

    for path in output_dir.glob("*.json"):
        path.unlink()

    print(f"done: {n_done} saved, {n_skip} skipped, {len(names)} total")


if __name__ == "__main__":
    main()
