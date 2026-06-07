#!/usr/bin/env python3
"""
金属板 PLY 场景 SAR FMCW 前向仿真。

管线：mesh 散射 → (a,τ) 路径 → FMCW ADC → (n_stops, 8, n_adc)。

运行（在 sar_sim 仓库根目录）::

  python simulate_pics.py

参数：
  --ply PATH          PLY 模型路径（默认 data/metal_plate_20x20x5cm.ply）
  --output-dir DIR    raw npz 输出目录（默认 output/raw_radar_data/）
  --scatter-mode MODE 散射模式：vertex | face_center | face_visible（默认 face_visible）
  --reflection        启用传送带平面 (z=0.5 m) 反射多径（默认关闭，仅直达径）
  --path-cache NPZ    缓存/读取静态 mesh 的 (a,τ)，加速重复仿真
  --skip-existing     输出 npz 已存在则跳过
  --start-angle-rad   第 1 停转台转角 (rad)，默认 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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
    adc_time_axis,
    angle_deg_at_stop,
    channel_tx_rx_index,
    rx_positions_at_stop,
    tx_positions_at_stop,
)
from sar_sim.config.scene import (
    DEFAULT_PLY_PATH,
    PLATE_X_HALF_M,
    raw_npz_stem,
    PLATE_Y_HALF_M,
    ReflectionConfig,
    ScatterMesh,
    ScatterMode,
    build_scatter_mesh,
    radar_output_dir,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
RAW_RADAR_DIR = radar_output_dir("raw_radar_data")
DEFAULT_OUTPUT_DIR = RAW_RADAR_DIR


# ---------------------------------------------------------------------------
# 路径配置与 (a, τ) 计算
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathConfig:
    scatter_mode: ScatterMode = ScatterMode.FACE_VISIBLE
    reflection: ReflectionConfig = ReflectionConfig()
    use_antenna_pattern: bool = True


def _channel_positions_at_stop(
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tx_all = tx_positions_at_stop(array, stop_index, rotation)
    rx_all = rx_positions_at_stop(array, stop_index, rotation)
    n_ch = array.n_channels
    tx_ch = np.zeros((n_ch, 3), dtype=np.float64)
    rx_ch = np.zeros((n_ch, 3), dtype=np.float64)
    for ch in range(n_ch):
        ti, ri = channel_tx_rx_index(ch, array)
        tx_ch[ch] = tx_all[ti]
        rx_ch[ch] = rx_all[ri]
    return tx_ch, rx_ch


def _require_cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("需要 NVIDIA GPU（PyTorch CUDA 不可用）")
    return torch.device("cuda")


def compute_paths_at_stop(
    mesh: ScatterMesh,
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig,
    *,
    path_config: PathConfig,
    radar: RadarConfig,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = _require_cuda_device()
    positions = torch.as_tensor(mesh.positions, dtype=torch.float64, device=device)
    area_w = torch.as_tensor(mesh.area_weight, dtype=torch.float64, device=device)
    normals = (
        torch.as_tensor(mesh.normals, dtype=torch.float64, device=device)
        if mesh.normals is not None
        else None
    )
    mode = path_config.scatter_mode if mesh.mode == ScatterMode.VERTEX else mesh.mode

    tx_ch, rx_ch = _channel_positions_at_stop(array, stop_index, rotation)
    tx_ch_t = torch.as_tensor(tx_ch, dtype=torch.float64, device=device)
    rx_ch_t = torch.as_tensor(rx_ch, dtype=torch.float64, device=device)
    antenna_angle_deg = angle_deg_at_stop(stop_index, rotation)

    n = positions.shape[0]
    n_ch = tx_ch_t.shape[0]
    refl = path_config.reflection
    n_refl = len(refl.points) if refl.enabled else 0
    n_path = 1 + n_refl

    diff_tx = positions[:, None, :] - tx_ch_t[None, :, :]
    r_tx = torch.linalg.norm(diff_tx, dim=2).clamp_min(1e-12)
    diff_rx = positions[:, None, :] - rx_ch_t[None, :, :]
    r_rx = torch.linalg.norm(diff_rx, dim=2).clamp_min(1e-12)

    if mode == ScatterMode.FACE_VISIBLE and normals is not None:
        inc = positions[:, None, :] - tx_ch_t[None, :, :]
        inc_len = torch.linalg.norm(inc, dim=2).clamp_min(1e-12)
        dot = torch.sum(normals[:, None, :] * inc, dim=2)
        vis = (dot > 0.0).to(torch.float64)
        cos_th = torch.abs(dot / inc_len) * vis
    else:
        vis = torch.ones((n, n_ch), dtype=torch.float64, device=device)
        cos_th = vis.clone()

    if radar.use_antenna_pattern:
        pattern = radar.antenna_pattern
        half_az = math.radians(pattern.hp_az_deg)
        half_el = math.radians(pattern.hp_el_deg)
        beta_az = pattern.az_beam_beta
        beta_el = pattern.el_beam_beta
        theta = math.radians(antenna_angle_deg)
        c_rot, s_rot = math.cos(theta), math.sin(theta)

        def _body_gain(diff):
            dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]
            dx_b = c_rot * dx + s_rot * dy
            dy_b = -s_rot * dx + c_rot * dy
            dz_b = dz
            ground = torch.sqrt(dx_b * dx_b + dz_b * dz_b)
            az = torch.atan2(dx_b, dz_b)
            el = torch.atan2(dy_b, ground)

            def axis_gain(angle, half_rad, beta):
                if half_rad <= 0:
                    return torch.zeros_like(angle)
                norm_a = torch.abs(angle) / half_rad
                x = torch.pow(norm_a, beta) * (math.pi / 4.0)
                out = torch.cos(x) ** 2
                return torch.where(x >= math.pi / 2.0, torch.zeros_like(out), out)

            return axis_gain(az, half_az, beta_az) * axis_gain(el, half_el, beta_el)

        gains = torch.sqrt(_body_gain(diff_tx) * _body_gain(diff_rx))
    else:
        gains = torch.ones((n, n_ch), dtype=torch.float64, device=device)

    a = torch.zeros((n, n_path, n_ch), dtype=torch.float64, device=device)
    tau = torch.zeros((n, n_path, n_ch), dtype=torch.float64, device=device)
    tau[:, 0, :] = (r_tx + r_rx) / c
    a[:, 0, :] = area_w[:, None] * cos_th * vis * gains / (r_tx * r_rx)

    if n_refl > 0:
        refl_pts = torch.as_tensor(refl.points, dtype=torch.float64, device=device)
        coeffs = torch.as_tensor(refl.coefficients, dtype=torch.float64, device=device)
        for ri in range(n_refl):
            rp = refl_pts[ri]
            d_pr = torch.linalg.norm(positions - rp, dim=1).clamp_min(1e-12)
            d_rr = torch.linalg.norm(rx_ch_t - rp, dim=1).clamp_min(1e-12)
            p_idx = 1 + ri
            tau[:, p_idx, :] = (r_tx + d_pr[:, None] + d_rr[None, :]) / c
            a[:, p_idx, :] = coeffs[ri] * area_w[:, None] * cos_th * vis * gains / (
                r_tx * d_pr[:, None] * d_rr[None, :]
            )

    return a.cpu().numpy(), tau.cpu().numpy()


# ---------------------------------------------------------------------------
# FMCW 合成
# ---------------------------------------------------------------------------


def synthesize_adc_from_paths(
    a: np.ndarray,
    tau: np.ndarray,
    radar: RadarConfig,
) -> np.ndarray:
    import torch

    if a.shape != tau.shape or a.ndim != 3:
        raise ValueError(f"期望 a/tau 同为 (N, n_path, n_ch)，得到 {a.shape} / {tau.shape}")

    device = _require_cuda_device()
    a_t = torch.as_tensor(a, dtype=torch.float64, device=device)
    tau_t = torch.as_tensor(tau, dtype=torch.float64, device=device)
    t = torch.as_tensor(adc_time_axis(radar), dtype=torch.float64, device=device)
    b = 2.0 * np.pi * radar.f_start_hz * tau_t - np.pi * radar.slope_hz_per_s * tau_t * tau_t
    c_coef = 2.0 * np.pi * radar.slope_hz_per_s * tau_t
    phase = b.unsqueeze(-1) + c_coef.unsqueeze(-1) * t
    data = torch.sum(a_t.unsqueeze(-1) * torch.exp(1j * phase), dim=(0, 1))
    return data.cpu().numpy().astype(np.complex64)


# ---------------------------------------------------------------------------
# (a, τ) 停点缓存
# ---------------------------------------------------------------------------


def _ply_digest(ply_path: str) -> str:
    if not ply_path:
        return ""
    p = Path(ply_path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def path_cache_meta(
    mesh: ScatterMesh,
    array: ArrayConfig,
    rotation: SarRotationConfig,
    path_config: PathConfig,
) -> dict[str, Any]:
    refl = path_config.reflection
    return {
        "ply_path": mesh.ply_path,
        "ply_digest": _ply_digest(mesh.ply_path),
        "scatter_mode": mesh.mode.value,
        "n_scatterers": mesh.n_scatterers,
        "reflection_enabled": refl.enabled,
        "reflection_points": list(refl.points),
        "reflection_coefficients": list(refl.coefficients),
        "n_stops": rotation.n_stops,
        "step_deg": rotation.step_deg,
        "start_angle_rad": rotation.start_angle_rad,
        "n_channels": array.n_channels,
    }


def path_cache_valid(cache_path: Path, meta: dict[str, Any]) -> bool:
    if not cache_path.is_file():
        return False
    meta_path = cache_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return False
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored == meta


def save_path_cache(
    cache_path: Path,
    a_all: np.ndarray,
    tau_all: np.ndarray,
    meta: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, a=a_all.astype(np.float64), tau=tau_all.astype(np.float64))
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_path_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(cache_path)
    return archive["a"], archive["tau"]


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    RAW_RADAR_DIR.mkdir(parents=True, exist_ok=True)


def _cuda_device_name() -> str:
    import torch

    _require_cuda_device()
    return torch.cuda.get_device_name(0)


def load_sar_cube(path: Path) -> tuple[np.ndarray, float]:
    archive = np.load(path)
    data = archive["data"]
    if data.ndim != 3 or data.shape[1] != 8:
        raise ValueError(f"期望 (n_stops, 8, n_adc)，得到 {data.shape}")
    start_angle_rad = (
        float(archive["start_angle_rad"]) if "start_angle_rad" in archive.files else 0.0
    )
    return np.asarray(data, dtype=np.complex64), start_angle_rad


def describe_scatter_mesh(mesh: ScatterMesh) -> str:
    xyz = mesh.positions
    return "\n".join(
        [
            f"scatter mesh @ {Path(mesh.ply_path).name if mesh.ply_path else '(targets)'}",
            f"  mode={mesh.mode.value}, scatterers={mesh.n_scatterers}",
            (
                f"  bbox x[{xyz[:, 0].min():.3f},{xyz[:, 0].max():.3f}] "
                f"y[{xyz[:, 1].min():.3f},{xyz[:, 1].max():.3f}] "
                f"z[{xyz[:, 2].min():.3f},{xyz[:, 2].max():.3f}]"
            ),
        ]
    )


def simulate_mesh_rotation_cube(
    mesh: ScatterMesh,
    *,
    radar: RadarConfig | None = None,
    array: ArrayConfig | None = None,
    rotation: SarRotationConfig | None = None,
    path_config: PathConfig | None = None,
    c: float = C_LIGHT,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    path_cache_path: Path | None = None,
) -> np.ndarray:
    radar = radar or RadarConfig()
    array = array or ArrayConfig()
    rotation = rotation or SarRotationConfig()
    path_config = path_config or PathConfig(scatter_mode=mesh.mode)

    meta = path_cache_meta(mesh, array, rotation, path_config)
    a_all = tau_all = None
    if path_cache_path is not None and path_cache_valid(path_cache_path, meta):
        a_all, tau_all = load_path_cache(path_cache_path)
        print(f"  path cache hit: {path_cache_path}", flush=True)

    cube = np.zeros((rotation.n_stops, array.n_channels, radar.n_adc_samples), dtype=np.complex64)

    if a_all is None:
        a_list: list[np.ndarray] = []
        tau_list: list[np.ndarray] = []
        for k in range(1, rotation.n_stops + 1):
            a_k, tau_k = compute_paths_at_stop(
                mesh, array, k, rotation, path_config=path_config, radar=radar, c=c
            )
            a_list.append(a_k)
            tau_list.append(tau_k)
            cube[k - 1] = synthesize_adc_from_paths(a_k, tau_k, radar)
            if progress_callback is not None:
                progress_callback(k, rotation.n_stops)
        if path_cache_path is not None:
            save_path_cache(path_cache_path, np.stack(a_list), np.stack(tau_list), meta)
            print(f"  path cache saved: {path_cache_path}", flush=True)
    else:
        for k in range(rotation.n_stops):
            cube[k] = synthesize_adc_from_paths(a_all[k], tau_all[k], radar)
            if progress_callback is not None:
                progress_callback(k + 1, rotation.n_stops)

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
    *,
    output_dir: Path,
    ply_path: Path | None = None,
    start_angle_rad: float = 0.0,
    scatter_mode: ScatterMode = ScatterMode.FACE_VISIBLE,
    reflection: ReflectionConfig | None = None,
    path_cache_path: Path | None = None,
) -> tuple[np.ndarray, Path, float]:
    ply = ply_path or DEFAULT_PLY_PATH
    mesh = build_scatter_mesh(ply, mode=scatter_mode)
    radar = RadarConfig()
    array = ArrayConfig()
    rotation = SarRotationConfig(start_angle_rad=start_angle_rad)
    refl = reflection or ReflectionConfig()
    path_config = PathConfig(scatter_mode=scatter_mode, reflection=refl)

    print(describe_scatter_mesh(mesh))
    if refl.enabled:
        print(f"  reflection: {len(refl.points)} points", flush=True)
    out_stem = raw_npz_stem(ply)
    print(f"    stops={rotation.n_stops}, output={out_stem}.npz", flush=True)

    def progress(k: int, n: int) -> None:
        if k == 1 or k == n or k % 100 == 0:
            print(f"    stop {k}/{n} ({angle_deg_at_stop(k, rotation):.1f} deg)", flush=True)

    t0 = time.perf_counter()
    cube = simulate_mesh_rotation_cube(
        mesh,
        radar=radar,
        array=array,
        rotation=rotation,
        path_config=path_config,
        progress_callback=progress,
        path_cache_path=path_cache_path,
    )
    path = save_raw_cube(cube, f"{out_stem}.npz", output_dir=output_dir, start_angle_rad=start_angle_rad)
    return cube, path, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="金属板 PLY 场景 SAR 仿真 → output/raw_radar_data/")
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scatter-mode",
        type=str,
        choices=[m.value for m in ScatterMode],
        default=ScatterMode.FACE_VISIBLE.value,
    )
    parser.add_argument("--reflection", action="store_true")
    parser.add_argument("--path-cache", type=Path, default=None, metavar="NPZ")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-angle-rad", type=float, default=0.0, metavar="RAD")
    args = parser.parse_args()

    if not (0.0 <= args.start_angle_rad < 2.0 * math.pi):
        raise SystemExit("start-angle-rad 须在 [0, 2π) 内")
    if not args.ply.is_file():
        raise SystemExit(f"PLY 不存在: {args.ply.resolve()}")

    output_dir = args.output_dir
    out_path = output_dir / f"{raw_npz_stem(args.ply)}.npz"
    scatter_mode = ScatterMode(args.scatter_mode)
    reflection = (
        ReflectionConfig.conveyor_belt_grid(
            x_range=(-PLATE_X_HALF_M - 0.05, PLATE_X_HALF_M + 0.05),
            y_range=(-PLATE_Y_HALF_M - 0.05, PLATE_Y_HALF_M + 0.05),
        )
        if args.reflection
        else ReflectionConfig()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"simulate metal_plate → {output_dir.resolve()}")
    print(f"  ply={args.ply.resolve()}, mode={scatter_mode.value}")
    print(f"  reflection={args.reflection}, gpu={_cuda_device_name()}")

    if args.skip_existing and out_path.is_file():
        print(f"skip (exists): {out_path}", flush=True)
        return

    cube, path, elapsed = simulate_one(
        output_dir=output_dir,
        ply_path=args.ply,
        start_angle_rad=args.start_angle_rad,
        scatter_mode=scatter_mode,
        reflection=reflection,
        path_cache_path=args.path_cache,
    )
    print(f"saved {path}  shape={cube.shape}  ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
