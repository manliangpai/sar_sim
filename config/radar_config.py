"""
毫米波雷达信号与接收配置：FMCW Chirp、ADC、天线方向图。

坐标系：主瓣 +Z，阵列沿 +X，y 横向；方向图在机体系中定义，随停点绕 +Z 转角旋转。
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
    IWR1843BOOST 单片天线 @ 78 GHz 方向图（TI SWRA758 Figure 10–12）。

    合理简化：FMCW 全 chirp 按 78 GHz 曲线；2TX+4RX 共 6 颗天线共用同一方向图。

    H 面（方位 az）：3 dB ±28°，6 dB ±50°
    E 面（俯仰 el）：3 dB ±14°，6 dB ±20°

    可分离模型 G(az, el) = G_az(az) * G_el(el)，单轴 cos^2 包络，
    指数 beta 由 3 dB / 6 dB 半角标定。
    """

    hp_az_deg: float = 28.0
    hp_el_deg: float = 14.0
    six_db_az_deg: float = 50.0
    six_db_el_deg: float = 20.0

    @classmethod
    def iwr1843_boost(cls) -> "AntennaPattern":
        return cls()

    @property
    def az_beam_beta(self) -> float:
        return _beamwidth_beta(self.hp_az_deg, self.six_db_az_deg)

    @property
    def el_beam_beta(self) -> float:
        return _beamwidth_beta(self.hp_el_deg, self.six_db_el_deg)


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


def rotate_delta_to_antenna_body(
    delta_xyz: np.ndarray,
    antenna_angle_deg: float,
) -> np.ndarray:
    """
    世界系视线向量 → 天线机体系（主瓣 +Z，阵列沿机体系 +X）。

    antenna_angle_deg 与 array.rotate_positions_about_z / angle_deg_at_stop 一致：
    停点阵列绕世界 +Z 转过该角后，将 delta 逆旋转 −θ 得到固连于 PCB 的坐标。
    """
    theta = math.radians(antenna_angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy, dz = float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, dz], dtype=np.float64)


def direction_angles_rad(
    antenna_xyz: np.ndarray,
    target_xyz: np.ndarray,
    antenna_angle_deg: float = 0.0,
) -> Tuple[float, float]:
    """机体系方位 atan2(dx,dz)，俯仰 atan2(dy, hypot(dx,dz))。"""
    delta = target_xyz - antenna_xyz
    db = rotate_delta_to_antenna_body(delta, antenna_angle_deg)
    dx, dy, dz = float(db[0]), float(db[1]), float(db[2])
    ground = math.hypot(dx, dz)
    return math.atan2(dx, dz), math.atan2(dy, ground)


def _beamwidth_beta(hp_deg: float, six_db_deg: float) -> float:
    """使 cos^2((θ/θ_3dB)^β · π/4) 在 θ_3dB、θ_6dB 处分别为 −3 dB、−6 dB。"""
    if hp_deg <= 0:
        raise ValueError("half-power angle must be positive")
    ratio = six_db_deg / hp_deg
    if ratio <= 1.0:
        raise ValueError("6 dB angle must exceed 3 dB half-power angle")
    return math.log(4.0 / 3.0) / math.log(ratio)


def _axis_gain(angle_rad: float, hp_deg: float, six_db_deg: float) -> float:
    if hp_deg <= 0:
        return 0.0
    beta = _beamwidth_beta(hp_deg, six_db_deg)
    hp_rad = math.radians(hp_deg)
    normalized = abs(angle_rad) / hp_rad
    if normalized == 0.0:
        return 1.0
    x = (normalized**beta) * (math.pi / 4.0)
    if x >= math.pi / 2.0:
        return 0.0
    return math.cos(x) ** 2


def pattern_gain_linear(
    antenna_xyz: np.ndarray,
    target_xyz: np.ndarray,
    pattern: AntennaPattern,
    antenna_angle_deg: float = 0.0,
) -> float:
    az, el = direction_angles_rad(
        antenna_xyz, target_xyz, antenna_angle_deg=antenna_angle_deg
    )
    g_az = _axis_gain(az, pattern.hp_az_deg, pattern.six_db_az_deg)
    g_el = _axis_gain(el, pattern.hp_el_deg, pattern.six_db_el_deg)
    return g_az * g_el


def monostatic_channel_gain(
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    target_xyz: np.ndarray,
    pattern: AntennaPattern,
    antenna_angle_deg: float = 0.0,
) -> Tuple[float, float, float]:
    """1T8R：sqrt(G_tx * G_rx) 幅度因子；TX/RX 共用同一停点转角。"""
    g_tx = pattern_gain_linear(
        tx_xyz, target_xyz, pattern, antenna_angle_deg=antenna_angle_deg
    )
    g_rx = pattern_gain_linear(
        rx_xyz, target_xyz, pattern, antenna_angle_deg=antenna_angle_deg
    )
    return math.sqrt(g_tx * g_rx), g_tx, g_rx
