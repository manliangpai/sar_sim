"""TI IWR1843-style mmWave SAR simulation package (2TX×4RX MIMO)."""

from sar_sim.config import (
    ArrayConfig,
    RadarConfig,
    SarRotationConfig,
    channel_tx_rx_index,
    PointTarget,
    rx_positions,
    rx_positions_at_stop,
    tx_positions,
    tx_positions_at_stop,
    two_corner_reflector_scene,
)
from sar_sim.simulate import simulate_sar_rotation_cube, save_raw_cube

__all__ = [
    "ArrayConfig",
    "SarRotationConfig",
    "RadarConfig",
    "PointTarget",
    "channel_tx_rx_index",
    "rx_positions",
    "rx_positions_at_stop",
    "tx_positions",
    "tx_positions_at_stop",
    "two_corner_reflector_scene",
    "simulate_sar_rotation_cube",
    "save_raw_cube",
]
