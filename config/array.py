"""
毫米波雷达阵列：2TX×4RX MIMO，绕 +Z 旋转合成孔径。

物理布局（停点 1，阵列在雷达平面 y=z=0，传送带在 z=0.5 m）：
  RX1–RX4：8 cm 起，沿 +X 每隔 1.9 mm
  TX1：RX4 后再 5.24 mm
  TX2：TX1 后再 7.6 mm

数据通道 8 路（2×4 MIMO），顺序：ch0–3 = TX1×RX1–4，ch4–7 = TX2×RX1–4。
仿真与后投影均按各通道真实的 TX、RX 双基地路径计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ArrayConfig:
    """2TX × 4RX，输出 8 路数据通道。"""

    n_tx: int = 2
    n_rx: int = 4
    n_channels: int = 8
    x0_m: float = 0.08  # RX1 的 x
    rx_step_m: float = 0.0019  # 1.9 mm
    tx1_gap_m: float = 0.00524  # RX4 → TX1
    tx2_gap_m: float = 0.0076  # TX1 → TX2


@dataclass(frozen=True)
class SarRotationConfig:
    """合成孔径停采：转台绕 +Z。"""

    n_stops: int = 600
    step_deg: float = 0.6
    start_angle_rad: float = 0.0
    """第 1 停阵列在 XOY 内绕 +Z 的转角 (rad)；0 表示停在 +X 方向（0 rad）。"""

    @property
    def total_rotation_deg(self) -> float:
        return (self.n_stops - 1) * self.step_deg


def angle_deg_at_stop(
    stop_index: int,
    rotation: SarRotationConfig | None = None,
) -> float:
    """停点 stop_index（1-based）相对 +X 的转角 (deg)。"""
    if stop_index < 1:
        raise ValueError("stop_index must be >= 1")
    rot = rotation or SarRotationConfig()
    return math.degrees(rot.start_angle_rad) + (stop_index - 1) * rot.step_deg


def rx_positions(array: ArrayConfig) -> np.ndarray:
    """(n_rx, 3) RX 坐标。"""
    xs = [array.x0_m + i * array.rx_step_m for i in range(array.n_rx)]
    ys = np.zeros(array.n_rx, dtype=np.float64)
    zs = np.zeros(array.n_rx, dtype=np.float64)
    return np.stack([xs, ys, zs], axis=1)


def tx_positions(array: ArrayConfig) -> np.ndarray:
    """(n_tx, 3) TX 坐标。"""
    rx4_x = array.x0_m + 3.0 * array.rx_step_m
    xs = [
        rx4_x + array.tx1_gap_m,
        rx4_x + array.tx1_gap_m + array.tx2_gap_m,
    ]
    ys = np.zeros(array.n_tx, dtype=np.float64)
    zs = np.zeros(array.n_tx, dtype=np.float64)
    return np.stack([xs, ys, zs], axis=1)


def channel_tx_rx_index(channel: int, array: ArrayConfig | None = None) -> tuple[int, int]:
    """数据通道 → (tx_idx, rx_idx)，均为 0-based。"""
    array = array or ArrayConfig()
    if channel < 0 or channel >= array.n_channels:
        raise ValueError(f"channel must be in [0, {array.n_channels})")
    return channel // array.n_rx, channel % array.n_rx


def rotate_positions_about_z(
    positions: np.ndarray,
    angle_deg: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    return ((positions - o) @ rot.T) + o


def rx_positions_at_stop(
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig | None = None,
) -> np.ndarray:
    if stop_index < 1:
        raise ValueError("stop_index must be >= 1")
    rot = rotation or SarRotationConfig()
    base = rx_positions(array)
    return rotate_positions_about_z(base, angle_deg_at_stop(stop_index, rot))


def tx_positions_at_stop(
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig | None = None,
) -> np.ndarray:
    if stop_index < 1:
        raise ValueError("stop_index must be >= 1")
    rot = rotation or SarRotationConfig()
    base = tx_positions(array)
    return rotate_positions_about_z(base, angle_deg_at_stop(stop_index, rot))
