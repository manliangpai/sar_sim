"""
毫米波雷达信号与接收配置：FMCW Chirp、ADC、天线方向图。

坐标系：主瓣 +Z，阵列沿 +X，y 横向。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

C_LIGHT = 299_709_000.0  # m/s


@dataclass(frozen=True)
class AntennaPattern:
    """
    IWR1843BOOST 近似方向图（TI SWRA758：方位 ±60°，俯仰 ±15°）。
    可分离 cos 包络，半功率角为 half_angle_*_deg。
    """

    half_angle_az_deg: float = 60.0
    half_angle_el_deg: float = 15.0
    cutoff_multiplier: float = 2.0

    @classmethod
    def iwr1843_boost(cls) -> "AntennaPattern":
        return cls(half_angle_az_deg=60.0, half_angle_el_deg=15.0)


@dataclass(frozen=True)
class RadarConfig:
    """TI IWR1843 风格 FMCW 与 ADC 参数。"""

    f_start_hz: float = 77e9
    slope_hz_per_s: float = 66.674e6 / 1e-6  # 66.674 MHz/us
    ramp_time_s: float = 59.99e-6  # 59.99 us
    adc_start_s: float = 5e-6  # 5 us
    adc_rate_hz: float = 4720e3  # 4720 ksps
    n_adc_samples: int = 256
    use_antenna_pattern: bool = True
    antenna_pattern: AntennaPattern = field(default_factory=AntennaPattern.iwr1843_boost)

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.f_start_hz

    @property
    def f_stop_hz(self) -> float:
        return self.f_start_hz + self.slope_hz_per_s * self.ramp_time_s

    @property
    def bandwidth_hz(self) -> float:
        return self.f_stop_hz - self.f_start_hz


def adc_time_axis(radar: RadarConfig) -> np.ndarray:
    """快时间采样轴 (s)。"""
    k = np.arange(radar.n_adc_samples, dtype=np.float64)
    return radar.adc_start_s + k / radar.adc_rate_hz


def direction_angles_rad(
    antenna_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> Tuple[float, float]:
    """方位 atan2(dx,dz)，俯仰 atan2(dy, hypot(dx,dz))。"""
    delta = target_xyz - antenna_xyz
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    ground = math.hypot(dx, dz)
    return math.atan2(dx, dz), math.atan2(dy, ground)


def _axis_gain(angle_rad: float, half_angle_deg: float, cutoff_multiplier: float) -> float:
    half_angle_rad = math.radians(half_angle_deg)
    if half_angle_rad <= 0:
        return 0.0
    normalized = abs(angle_rad) / half_angle_rad
    if normalized >= cutoff_multiplier:
        return 0.0
    return math.cos(math.pi * normalized / 4.0) ** 2


def pattern_gain_linear(
    antenna_xyz: np.ndarray,
    target_xyz: np.ndarray,
    pattern: AntennaPattern,
) -> float:
    az, el = direction_angles_rad(antenna_xyz, target_xyz)
    g_az = _axis_gain(az, pattern.half_angle_az_deg, pattern.cutoff_multiplier)
    g_el = _axis_gain(el, pattern.half_angle_el_deg, pattern.cutoff_multiplier)
    return g_az * g_el


def monostatic_channel_gain(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    target_xyz: np.ndarray,
    pattern: AntennaPattern,
) -> Tuple[float, float, float]:
    """1T8R：sqrt(G_tx * G_rx) 幅度因子。"""
    g_tx = pattern_gain_linear(tx_xyz, target_xyz, pattern)
    g_rx = pattern_gain_linear(rx_xyz, target_xyz, pattern)
    return math.sqrt(g_tx * g_rx), g_tx, g_rx
