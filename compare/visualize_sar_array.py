#!/usr/bin/env python3
"""
合成孔径 2TX×4RX 阵列 3D 可视化：全停点轨迹 + TX2 在指定停点的 3 dB 视野。

运行（在 sar_sim 仓库根目录）::

  python compare/visualize_sar_array.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sar_sim.config.array import (
    ArrayConfig,
    SarRotationConfig,
    angle_deg_at_stop,
    rx_positions_at_stop,
    tx_positions_at_stop,
)
from sar_sim.config.radar_config import AntennaPattern
from sar_sim.config.scene import PLATE_Z_MAX_M

DEFAULT_FOV_STOPS = (1, 151)
_PLANE_Z = 0.5
_XY_HALF = 0.2
_PAIR_ANGLE_DEG = 5.0

_COLOR_TX = "#e74c3c"
_COLOR_RX = "#3498db"
_COLOR_BORESIGHT = "#f39c12"
_COLOR_TX2_STOP1 = "#ff7f0e"
_COLOR_TX2_STOP151 = "#8e44ad"
_COLOR_RAND_A = "#16a085"
_COLOR_RAND_B = "#2c3e50"

_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
)


def _setup_matplotlib() -> str:
    plt.rcParams["axes.unicode_minus"] = False
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in _FONT_CANDIDATES:
        if name in available:
            chosen = name
            break
    else:
        for path in (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
        ):
            if path.is_file():
                fm.fontManager.addfont(str(path))
                chosen = fm.FontProperties(fname=str(path)).get_name()
                break
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    return chosen


def _body_delta_from_az_el(az: float, el: float, range_m: float) -> np.ndarray:
    ce = math.cos(el)
    return np.array(
        [range_m * ce * math.sin(az), range_m * math.sin(el), range_m * ce * math.cos(az)],
        dtype=np.float64,
    )


def _body_delta_to_world(
    delta_body: np.ndarray,
    antenna_angle_deg: float,
) -> np.ndarray:
    theta = math.radians(antenna_angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy, dz = delta_body
    return np.array([c * dx - s * dy, s * dx + c * dy, dz], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def sample_plane_point_pair(
    *,
    z: float = _PLANE_Z,
    xy_half: float = _XY_HALF,
    separation_deg: float = _PAIR_ANGLE_DEG,
    max_tries: int = 50_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 z=z、|x|,|y|≤xy_half 内随机取两点，使原点→两点的夹角为 separation_deg。
    """
    rng = rng or np.random.default_rng()
    theta = math.radians(separation_deg)

    for _ in range(max_tries):
        p1 = np.array(
            [rng.uniform(-xy_half, xy_half), rng.uniform(-xy_half, xy_half), z],
            dtype=np.float64,
        )
        u1 = _unit(p1)
        ref = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = _unit(np.cross(u1, ref))
        w2 = _unit(np.cross(u1, w))
        phi = rng.uniform(0.0, 2.0 * math.pi)
        v2_dir = math.cos(theta) * u1 + math.sin(theta) * (
            math.cos(phi) * w + math.sin(phi) * w2
        )
        if v2_dir[2] <= 1e-6:
            continue
        p2 = (z / v2_dir[2]) * v2_dir
        if abs(p2[0]) <= xy_half and abs(p2[1]) <= xy_half:
            return p1, p2

    raise RuntimeError(
        f"{max_tries} 次尝试未找到满足条件的点对；可增大平面范围或略减小夹角。"
    )


def _draw_random_pair(ax, p1: np.ndarray, p2: np.ndarray) -> None:
    origin = np.zeros(3)
    for p, c, label in ((p1, _COLOR_RAND_A, "A"), (p2, _COLOR_RAND_B, "B")):
        ax.plot(
            [origin[0], p[0]], [origin[1], p[1]], [origin[2], p[2]],
            color=c, lw=2.0, ls="--", alpha=0.9,
        )
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            c=c, s=70, marker="o", depthshade=True, edgecolors="k", linewidths=0.4,
            zorder=6,
        )
        ax.text(p[0], p[1], p[2], f"  {label}", fontsize=9, color=c)


def _tx2_at_stop(
    array: ArrayConfig,
    stop_index: int,
    rotation: SarRotationConfig,
) -> np.ndarray:
    return tx_positions_at_stop(array, stop_index, rotation)[1]


def fov_summary_line(
    pattern: AntennaPattern,
    range_m: float,
    stop_index: int,
    angle_deg: float,
) -> str:
    w_az = 2.0 * range_m * math.tan(math.radians(pattern.hp_az_deg))
    w_el = 2.0 * range_m * math.tan(math.radians(pattern.hp_el_deg))
    return (
        f"  停点 {stop_index:3d}  转角 {angle_deg:6.2f}°  "
        f"@ z={range_m:.2f} m 截面约 {w_az * 1e2:.0f}×{w_el * 1e2:.0f} cm"
    )


def _fov_rim_loop_body(
    pattern: AntennaPattern,
    range_m: float,
    *,
    n_per_edge: int = 16,
) -> np.ndarray:
    """
    3 dB 矩形角域 (|az|≤hp_az, |el|≤hp_el) 的口沿闭合折线（机体系）。
    须走四条边，不能用边中点 (±az,0)/(0,±el) 否则三角面会自交。
    """
    hp_az = math.radians(pattern.hp_az_deg)
    hp_el = math.radians(pattern.hp_el_deg)
    az_lo, az_hi = -hp_az, hp_az
    el_lo, el_hi = -hp_el, hp_el
    pts: list[np.ndarray] = []

    for az in np.linspace(az_lo, az_hi, n_per_edge):
        pts.append(_body_delta_from_az_el(float(az), el_lo, range_m))
    for el in np.linspace(el_lo, el_hi, n_per_edge)[1:]:
        pts.append(_body_delta_from_az_el(az_hi, float(el), range_m))
    for az in np.linspace(az_hi, az_lo, n_per_edge)[1:]:
        pts.append(_body_delta_from_az_el(float(az), el_hi, range_m))
    for el in np.linspace(el_hi, el_lo, n_per_edge)[1:-1]:
        pts.append(_body_delta_from_az_el(az_lo, float(el), range_m))

    return np.stack(pts, axis=0)


def _fov_rim_world(
    apex: np.ndarray,
    antenna_angle_deg: float,
    pattern: AntennaPattern,
    range_m: float,
) -> np.ndarray:
    apex = np.asarray(apex, dtype=np.float64)
    loop_body = _fov_rim_loop_body(pattern, range_m)
    return np.stack(
        [_body_delta_to_world(p, antenna_angle_deg) + apex for p in loop_body],
        axis=0,
    )


def _fov_pyramid_faces(
    apex: np.ndarray,
    antenna_angle_deg: float,
    pattern: AntennaPattern,
    range_m: float,
) -> list[np.ndarray]:
    """从 apex 对口沿三角扇形铺面，保证无自交。"""
    apex = np.asarray(apex, dtype=np.float64)
    rim = _fov_rim_world(apex, antenna_angle_deg, pattern, range_m)
    faces = []
    n = len(rim)
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([apex, rim[i], rim[j]]))
    return faces


def _draw_filled_pyramid(
    ax,
    apex: np.ndarray,
    antenna_angle_deg: float,
    pattern: AntennaPattern,
    range_m: float,
    color: str,
    *,
    alpha: float = 0.22,
) -> None:
    rim = _fov_rim_world(apex, antenna_angle_deg, pattern, range_m)
    faces = _fov_pyramid_faces(apex, antenna_angle_deg, pattern, range_m)
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            linewidths=0.3,
        )
    )
    ax.plot(
        np.append(rim[:, 0], rim[0, 0]),
        np.append(rim[:, 1], rim[0, 1]),
        np.append(rim[:, 2], rim[0, 2]),
        color=color,
        lw=1.3,
        alpha=0.95,
    )


def _all_antenna_positions(
    array: ArrayConfig,
    rotation: SarRotationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = rotation.n_stops
    all_tx = np.zeros((n, array.n_tx, 3), dtype=np.float64)
    all_rx = np.zeros((n, array.n_rx, 3), dtype=np.float64)
    for k in range(1, n + 1):
        all_tx[k - 1] = tx_positions_at_stop(array, k, rotation)
        all_rx[k - 1] = rx_positions_at_stop(array, k, rotation)
    return all_tx, all_rx


def build_figure(
    array: ArrayConfig,
    rotation: SarRotationConfig,
    pattern: AntennaPattern,
    *,
    fov_range_m: float,
    fov_stops: tuple[int, ...],
    plane_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> plt.Figure:
    for k in fov_stops:
        if not (1 <= k <= rotation.n_stops):
            raise ValueError(f"停点须在 [1, {rotation.n_stops}]，得到 {k}")

    stop_colors = {
        1: _COLOR_TX2_STOP1,
        151: _COLOR_TX2_STOP151,
    }

    all_tx, all_rx = _all_antenna_positions(array, rotation)
    flat_tx = all_tx.reshape(-1, 3)
    flat_rx = all_rx.reshape(-1, 3)
    xy = np.vstack([flat_tx[:, :2], flat_rx[:, :2]])

    span = max(0.15, float(np.max(np.linalg.norm(xy, axis=1))) * 1.12)
    if plane_pair is not None:
        p1, p2 = plane_pair
        span = max(span, float(np.max(np.abs(np.vstack([p1[:2], p2[:2]])))) * 1.15)
    span = max(span, _XY_HALF * 1.1)
    for k in fov_stops:
        apex = _tx2_at_stop(array, k, rotation)
        r = float(np.hypot(apex[0], apex[1]))
        span = max(span, fov_range_m * math.tan(math.radians(pattern.hp_az_deg)) * 1.2 + r)

    fig = plt.figure(figsize=(10, 9))
    fig.canvas.manager.set_window_title("SAR Array — TX2 FOV @ stops 1 & 151")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")

    ax.scatter(
        flat_tx[:, 0], flat_tx[:, 1], flat_tx[:, 2],
        c=_COLOR_TX, s=10, marker="^", alpha=0.35, depthshade=False,
    )
    ax.scatter(
        flat_rx[:, 0], flat_rx[:, 1], flat_rx[:, 2],
        c=_COLOR_RX, s=8, marker="s", alpha=0.35, depthshade=False,
    )

    for k in fov_stops:
        apex = _tx2_at_stop(array, k, rotation)
        ang = angle_deg_at_stop(k, rotation)
        color = stop_colors.get(k, _COLOR_TX2_STOP1)
        ax.scatter(
            [apex[0]], [apex[1]], [apex[2]],
            c=color, s=90, marker="^", depthshade=True, edgecolors="k", linewidths=0.4,
            zorder=5,
        )
        _draw_filled_pyramid(ax, apex, ang, pattern, fov_range_m, color, alpha=0.22)

    if plane_pair is not None:
        _draw_random_pair(ax, plane_pair[0], plane_pair[1])

    ax.quiver(0, 0, 0, 0, 0, 0.1, color=_COLOR_BORESIGHT, arrow_length_ratio=0.15, linewidth=2)

    legend_handles = [
        Line2D(
            [], [], linestyle="None", marker="^", markersize=8,
            markerfacecolor=_COLOR_TX, markeredgecolor=_COLOR_TX,
            label="TX（各停点）",
        ),
        Line2D(
            [], [], linestyle="None", marker="s", markersize=7,
            markerfacecolor=_COLOR_RX, markeredgecolor=_COLOR_RX,
            label="RX（各停点）",
        ),
        Line2D([], [], color=_COLOR_BORESIGHT, lw=2, label="+Z 主瓣"),
        Line2D([], [], color=_COLOR_TX2_STOP1, lw=2, label="3 dB TX2 @ 停点 1"),
        Line2D([], [], color=_COLOR_TX2_STOP151, lw=2, label="3 dB TX2 @ 停点 151"),
    ]
    if plane_pair is not None:
        legend_handles.extend(
            [
                Line2D([], [], color=_COLOR_RAND_A, lw=2, ls="--", label="随机点 A"),
                Line2D([], [], color=_COLOR_RAND_B, lw=2, ls="--", label="随机点 B"),
                Line2D(
                    [], [], color="#7f8c8d", lw=0,
                    label=f"∠AOB={_PAIR_ANGLE_DEG:g}° (z={_PLANE_Z} m)",
                ),
            ]
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.92)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-0.02, fov_range_m * 1.08)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1, 1, 0.5))

    return fig


def main() -> None:
    _setup_matplotlib()
    parser = argparse.ArgumentParser(description="SAR 阵列 3D 可视化")
    parser.add_argument("--stops", type=int, default=None)
    parser.add_argument("--step-deg", type=float, default=None)
    parser.add_argument("--start-angle-deg", type=float, default=0.0)
    parser.add_argument("--fov-range", type=float, default=PLATE_Z_MAX_M)
    parser.add_argument(
        "--fov-stops",
        type=int,
        nargs="+",
        default=list(DEFAULT_FOV_STOPS),
        help="绘制 TX2 3 dB 视野的停点（默认 1 151）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机点对种子（默认每次不同）")
    args = parser.parse_args()

    pattern = AntennaPattern.iwr1843_boost()
    array = ArrayConfig()
    rot_kw: dict = {"start_angle_rad": math.radians(args.start_angle_deg)}
    if args.stops is not None:
        rot_kw["n_stops"] = args.stops
    if args.step_deg is not None:
        rot_kw["step_deg"] = args.step_deg
    rotation = SarRotationConfig(**rot_kw)

    fov_stops = tuple(sorted(set(args.fov_stops)))
    for k in fov_stops:
        if k > rotation.n_stops:
            raise SystemExit(f"停点 {k} 超过总停数 {rotation.n_stops}")

    hp_az, hp_el = pattern.hp_az_deg, pattern.hp_el_deg
    print(f"TX2  3 dB：方位 ±{hp_az:.0f}°，俯仰 ±{hp_el:.0f}°")
    for k in fov_stops:
        print(fov_summary_line(pattern, args.fov_range, k, angle_deg_at_stop(k, rotation)))
    print()
    rng = np.random.default_rng(args.seed)
    p1, p2 = sample_plane_point_pair(rng=rng)
    ang = _angle_between_deg(p1, p2)
    print(
        f"随机点对 (z={_PLANE_Z} m, |x|,|y|≤{_XY_HALF} m): "
        f"A=({p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f})  "
        f"B=({p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f})  "
        f"夹角={ang:.4f}°"
    )
    print(f"{array.n_tx}TX x {array.n_rx}RX, {rotation.n_stops} stops.")
    build_figure(
        array,
        rotation,
        pattern,
        fov_range_m=args.fov_range,
        fov_stops=fov_stops,
        plane_pair=(p1, p2),
    )
    plt.show()


if __name__ == "__main__":
    main()
