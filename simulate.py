#!/usr/bin/env python3
"""
在阵列、雷达、场景配置下生成合成孔径 FMCW 原始数据立方体。

2TX×4RX：每个停点先 TX1 发 1 个 chirp（4 RX 同时采），再 TX2 发 1 个 chirp。
输出 (n_stops, 8, n_adc)：ch0–3=TX1·RX1–4，ch4–7=TX2·RX1–4。

场景几何见 ``sar_sim.config.scene``；本脚本只整合
``RadarConfig`` + ``ArrayConfig`` + ``SarRotationConfig`` + 散射体列表。

运行示例（在仓库根目录执行）::

  python sar_sim/simulate.py    # 两角反（默认）
  python sar_sim/simulate.py --scene two_corners
  python sar_sim/simulate.py --scene concave_room
  python sar_sim/simulate.py --scene concave_room --wall-spacing 0.1 --add-two-corner-reflectors
  python sar_sim/simulate.py --start-angle-rad 0.5236   # 第 1 停 30°（π/6 rad）
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parents[1]
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
    add_scene_cli_arguments,
    angle_deg_at_stop,
    describe_scene,
    monostatic_channel_gain,
    rx_positions,
    rx_positions_at_stop,
    scene_from_cli,
    scene_output_filename,
    targets_to_xyz,
    tx_positions,
    tx_positions_at_stop,
    two_corner_reflector_scene,
)

# 后处理脚本默认输入（两角反场景）
DEFAULT_OUTPUT_NAME = scene_output_filename("two_corners")

_PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = _PACKAGE_ROOT / "output"
RAW_RADAR_DIR = OUTPUT_ROOT / "raw_radar_data"
PROCESSED_RADAR_DIR = OUTPUT_ROOT / "processed_radar_data"


def ensure_output_dirs() -> None:
    RAW_RADAR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RADAR_DIR.mkdir(parents=True, exist_ok=True)


def _simulate_channels_for_one_tx(
    radar: RadarConfig,
    array: ArrayConfig,
    targets: Sequence[PointTarget],
    tx_pos: np.ndarray,
    rx_positions: np.ndarray,
    c: float = C_LIGHT,
) -> np.ndarray:
    """
    单个 TX 发一个 chirp，4 个 RX 同时接收。

    返回 (n_rx, n_adc_samples) complex64。
    双基地：tau = (|TX-目标| + |目标-RX|) / c。
    """
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
    """一个停点：TX1 chirp + TX2 chirp → (8, n_adc)。"""
    tx_pos = tx_positions_at_stop(array, stop_index, rotation)
    rx_pos = rx_positions_at_stop(array, stop_index, rotation)
    frame = np.zeros((array.n_channels, radar.n_adc_samples), dtype=np.complex64)

    for tx_i in range(array.n_tx):
        block = _simulate_channels_for_one_tx(
            radar, array, targets, tx_pos[tx_i], rx_pos, c=c
        )
        frame[tx_i * array.n_rx : (tx_i + 1) * array.n_rx, :] = block

    return frame


def simulate_sar_rotation_cube(
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    targets: Sequence[PointTarget] | None = None,
    c: float = C_LIGHT,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """停采转台 SAR：返回 (n_stops, 8, n_adc_samples) complex64。"""
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()
    target_list = list(targets if targets is not None else two_corner_reflector_scene())

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
    ensure_output_dirs()
    out_dir = output_dir or RAW_RADAR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    np.savez_compressed(
        path,
        data=data.astype(np.complex64, copy=False),
        start_angle_rad=np.float64(start_angle_rad),
    )
    return path


def run_simulation(
    targets: Sequence[PointTarget],
    *,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    output_filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, Path]:
    """整合三类配置并采集，返回 (cube, 保存路径)。"""
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()
    cube = simulate_sar_rotation_cube(
        radar=radar,
        array=array,
        rotation=rotation,
        targets=targets,
        progress_callback=progress_callback,
    )
    path = save_raw_cube(
        cube, output_filename, start_angle_rad=rotation.start_angle_rad
    )
    return cube, path


def _validate_start_angle_rad(value: float) -> float:
    if not (0.0 <= value < 2.0 * math.pi):
        raise argparse.ArgumentTypeError(
            f"start_angle_rad 须在 [0, 2π) rad 内，得到 {value}"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAR FMCW 原始数据：雷达 + 阵列 + 转台 + 场景"
    )
    add_scene_cli_arguments(parser)
    parser.add_argument(
        "--start-angle-rad",
        type=_validate_start_angle_rad,
        default=0.0,
        metavar="RAD",
        help="第 1 停转角 (rad)，0=停在 +X；范围 [0, 2π)，默认 0",
    )
    args = parser.parse_args()

    radar = RadarConfig()
    array = ArrayConfig()
    rotation = SarRotationConfig(start_angle_rad=args.start_angle_rad)
    scene_name, targets, out_name = scene_from_cli(args)

    print("SAR simulation (sar_sim) — 2TX × 4RX")
    print(f"  radar: {radar.f_start_hz/1e9:.0f} GHz, {radar.n_adc_samples} ADC")
    print(f"  array: 2 TX, 4 RX → 8 ch; RX step {array.rx_step_m*1e3:.2f} mm")
    print(f"  SAR: {rotation.n_stops} stops × 2 chirps/stop")
    print(
        f"  start_angle_rad: {rotation.start_angle_rad:.6g} "
        f"({math.degrees(rotation.start_angle_rad):.4g} deg)"
    )
    print(describe_scene(scene_name, targets))
    print(f"  stop-1 TX1 (m): {tx_positions_at_stop(array, 1, rotation)[0]}")
    print(f"  stop-1 RX1 (m): {rx_positions_at_stop(array, 1, rotation)[0]}")

    t0 = time.perf_counter()

    def progress(k: int, n: int) -> None:
        if k == 1 or k == n or k % 50 == 0:
            print(f"  stop {k}/{n}  ({angle_deg_at_stop(k, rotation):.2f} deg)")

    cube, path = run_simulation(
        targets,
        radar=radar,
        array=array,
        rotation=rotation,
        output_filename=out_name,
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - t0

    print(f"Saved {path}")
    print(f"  shape: {cube.shape}, dtype: {cube.dtype}")
    print(f"  |data| peak: {np.abs(cube).max():.4g}")
    print(f"  elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
