"""TI IWR1843-style mmWave SAR simulation package (2TX×4RX MIMO)."""

from sar_sim.config import (
    ArrayConfig,
    RadarConfig,
    SarRotationConfig,
    channel_tx_rx_index,
    PointTarget,
    pixel_pattern_scene,
    rx_positions,
    rx_positions_at_stop,
    tx_positions,
    tx_positions_at_stop,
)
from sar_sim.simulate_pics import load_sar_cube, save_raw_cube, simulate_sar_rotation_cube

__all__ = [
    "ArrayConfig",
    "SarRotationConfig",
    "RadarConfig",
    "PointTarget",
    "channel_tx_rx_index",
    "pixel_pattern_scene",
    "rx_positions",
    "rx_positions_at_stop",
    "tx_positions",
    "tx_positions_at_stop",
    "load_sar_cube",
    "simulate_sar_rotation_cube",
    "save_raw_cube",
]
