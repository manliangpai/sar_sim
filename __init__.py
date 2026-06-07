"""TI IWR1843-style mmWave SAR simulation package (2TX×4RX MIMO)."""

from sar_sim.config import (
    ArrayConfig,
    RadarConfig,
    SarRotationConfig,
    ScatterMesh,
    build_scatter_mesh,
    channel_tx_rx_index,
    rx_positions,
    rx_positions_at_stop,
    tx_positions,
    tx_positions_at_stop,
)
from sar_sim.simulate_pics import load_sar_cube, save_raw_cube, simulate_mesh_rotation_cube

__all__ = [
    "ArrayConfig",
    "SarRotationConfig",
    "RadarConfig",
    "ScatterMesh",
    "build_scatter_mesh",
    "channel_tx_rx_index",
    "rx_positions",
    "rx_positions_at_stop",
    "tx_positions",
    "tx_positions_at_stop",
    "load_sar_cube",
    "simulate_mesh_rotation_cube",
    "save_raw_cube",
]
