#!/usr/bin/env python3
"""
双点散射体对比实验：随机 5° 夹角两点 → SAR 仿真 (600,8,256) → z=0.5 m BP 成像。

不写入 raw/processed 文件；不修改项目内其它 .py。

运行（在 sar_sim 仓库根目录）::

  python compare/simulate_compare.py          # 3D 场景 + BP 成像（两个窗口）
  python compare/simulate_compare.py --gpu
  python compare/simulate_compare.py --no-plot   # 仅终端输出
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _repo_parent = Path(__file__).resolve().parents[2]
    if str(_repo_parent) not in sys.path:
        sys.path.insert(0, str(_repo_parent))

import numpy as np

from sar_sim.config import ArrayConfig, PointTarget, RadarConfig, SarRotationConfig
from sar_sim.config import C_LIGHT
from sar_sim.config.radar_config import AntennaPattern
from sar_sim.simulate_pics import simulate_sar_rotation_cube

_DEFAULT_FOV_STOPS = (1, 151)

# ---------------------------------------------------------------------------
# 成像网格（本脚本专用：步长 0.02 m）
# ---------------------------------------------------------------------------
BP_Z_M = 0.5
BP_XY_MIN_M = -0.3
BP_XY_MAX_M = 0.3
BP_STEP_M = 0.005
BP_GRID_N = int(round((BP_XY_MAX_M - BP_XY_MIN_M) / BP_STEP_M))  # 20

_PAIR_Z = 0.5
_PAIR_XY_HALF = 0.2
_PAIR_ANGLE_DEG = 5.0


def _load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_viz_array():
    """compare/visualize_sar_array.py：点对采样 + 3D 阵列场景。"""
    path = Path(__file__).resolve().parent / "visualize_sar_array.py"
    return _load_module_from_path(path, "viz_sar_array_cmp")


def _load_viz_z_plane():
    """仓库根 visualize_on_a_z_plane.py：成像图样式。"""
    path = Path(__file__).resolve().parents[1] / "visualize_on_a_z_plane.py"
    return _load_module_from_path(path, "viz_z_plane_cmp")


def bp_grid_axes() -> tuple[np.ndarray, np.ndarray]:
    """20×20 格心：x/y ∈ [-0.2,0.2] m，步长 0.02 m。"""
    axis = (
        BP_XY_MIN_M + (np.arange(BP_GRID_N, dtype=np.float64) + 0.5) * BP_STEP_M
    ).astype(np.float64)
    return axis, axis.copy()


def targets_from_plane_points(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    amplitude: float = 1.0,
) -> list[PointTarget]:
    return [
        PointTarget(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]), amplitude=amplitude),
        PointTarget(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]), amplitude=amplitude),
    ]


def simulate_raw_cube(
    targets: list[PointTarget],
    *,
    use_gpu: bool,
    start_angle_rad: float = 0.0,
) -> np.ndarray:
    """(n_stops, 8, n_adc) complex64，默认 600×8×256。"""
    radar = RadarConfig()
    array = ArrayConfig()
    rotation = SarRotationConfig(start_angle_rad=start_angle_rad)

    def progress(k: int, n: int) -> None:
        if k == 1 or k == n or k % 100 == 0:
            print(f"  simulate stop {k}/{n}", flush=True)

    cube = simulate_sar_rotation_cube(
        radar=radar,
        array=array,
        rotation=rotation,
        targets=targets,
        c=C_LIGHT,
        progress_callback=progress,
        use_gpu=use_gpu,
    )
    expected = (rotation.n_stops, array.n_channels, radar.n_adc_samples)
    if cube.shape != expected:
        raise RuntimeError(f"期望 {expected}，得到 {cube.shape}")
    return cube


def backproject_z_plane(
    raw_cube: np.ndarray,
    *,
    z_m: float = BP_Z_M,
    start_angle_rad: float = 0.0,
) -> np.ndarray:
    """距离 FFT + z 平面 BP → (20, 20) complex。"""
    from sar_sim.process_pics import (
        _backproject_z_plane_gpu,
        _range_fft_cube_gpu,
        _require_cuda_device,
    )

    rotation = SarRotationConfig(
        n_stops=raw_cube.shape[0],
        start_angle_rad=start_angle_rad,
    )
    radar = RadarConfig()
    array = ArrayConfig()
    x_axis, y_axis = bp_grid_axes()

    device = _require_cuda_device()
    t0 = time.perf_counter()
    range_cube = _range_fft_cube_gpu(raw_cube, device)
    t_fft = time.perf_counter() - t0

    t1 = time.perf_counter()
    image = _backproject_z_plane_gpu(
        range_cube,
        z_m=z_m,
        x_axis=x_axis,
        y_axis=y_axis,
        radar=radar,
        array=array,
        rotation=rotation,
        device=device,
    )
    t_bp = time.perf_counter() - t1

    if image.shape != (BP_GRID_N, BP_GRID_N):
        raise RuntimeError(f"BP 形状须 ({BP_GRID_N},{BP_GRID_N})，得到 {image.shape}")

    print(f"  range FFT: {t_fft:.1f}s  BP: {t_bp:.1f}s  grid {BP_GRID_N}×{BP_GRID_N}")
    return image.astype(np.complex64)


def _top_peaks(mag: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, n: int = 4) -> list[tuple[float, float, float]]:
    """按幅度取前 n 个像素（对照 GT 点大致位置）。"""
    flat = mag.ravel()
    if flat.size == 0:
        return []
    order = np.argsort(flat)[::-1][:n]
    out: list[tuple[float, float, float]] = []
    for idx in order:
        r, c = divmod(int(idx), mag.shape[1])
        out.append((float(x_axis[c]), float(y_axis[r]), float(flat[idx])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="双点 5° 场景：仿真 + z=0.5 BP（无文件输出）")
    parser.add_argument("--seed", type=int, default=None, help="随机点对种子")
    parser.add_argument("--gpu", action="store_true", help="仿真使用 GPU（BP 始终用 GPU）")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不弹窗（默认显示 3D 场景 + BP 成像两个窗口）",
    )
    parser.add_argument("--save", type=Path, default=None, help="可选：保存 BP npz 到该路径")
    parser.add_argument("--amplitude", type=float, default=1.0, help="点散射体幅度")
    args = parser.parse_args()

    viz_arr = _load_viz_array()
    sample_pair = viz_arr.sample_plane_point_pair
    angle_between = viz_arr._angle_between_deg
    rng = np.random.default_rng(args.seed)
    p1, p2 = sample_pair(
        z=_PAIR_Z,
        xy_half=_PAIR_XY_HALF,
        separation_deg=_PAIR_ANGLE_DEG,
        rng=rng,
    )
    ang = angle_between(p1, p2)
    targets = targets_from_plane_points(p1, p2, amplitude=args.amplitude)

    print("场景：两点散射体（平面 z=0.5 m，|x|,|y|≤0.2 m，∠AOB=5°）")
    print(f"  A = ({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f})")
    print(f"  B = ({p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f})  夹角 {ang:.4f}°")
    print(f"  幅度 = {args.amplitude}")

    t0 = time.perf_counter()
    print("1) SAR 仿真 …")
    raw = simulate_raw_cube(targets, use_gpu=args.gpu)
    print(f"   raw shape {raw.shape}  dtype {raw.dtype}  (不保存文件)")

    print("2) 距离 FFT + z=0.5 m BP …")
    print(
        f"   网格 x/y ∈ [{BP_XY_MIN_M}, {BP_XY_MAX_M}] m, "
        f"步长 {BP_STEP_M} m → {BP_GRID_N}×{BP_GRID_N}"
    )
    image = backproject_z_plane(raw)
    mag = np.abs(image)
    peak = float(mag.max())
    x_ax, y_ax = bp_grid_axes()
    peaks = _top_peaks(mag, x_ax, y_ax)

    elapsed = time.perf_counter() - t0
    print(f"3) 结果  peak={peak:.4g}  耗时 {elapsed:.1f}s")
    if peaks:
        print("   BP 局部峰 (x,y,m) [m]:")
        for x, y, v in peaks[:4]:
            print(f"     ({x:+.3f}, {y:+.3f})  mag={v:.4g}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        data = np.stack([image.real, image.imag], axis=-1).astype(np.float32)
        np.savez_compressed(
            args.save,
            data=data,
            magnitude=mag.astype(np.float32),
            point_a=p1,
            point_b=p2,
            x_axis=x_ax,
            y_axis=y_ax,
        )
        print(f"   已保存 {args.save.resolve()}")

    if not args.no_plot:
        show_result_figures(
            viz_arr,
            p1=p1,
            p2=p2,
            angle_deg=ang,
            image=image,
            x_axis=x_ax,
            y_axis=y_ax,
        )


def show_result_figures(
    viz_arr,
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    angle_deg: float,
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> None:
    """两个交互窗口：3D 阵列+散射点；z 平面 BP（样式同 visualize_on_a_z_plane）。"""
    import matplotlib.pyplot as plt

    viz_arr._setup_matplotlib()
    _load_viz_z_plane().setup_chinese_font()

    array = ArrayConfig()
    rotation = SarRotationConfig()
    pattern = AntennaPattern.iwr1843_boost()

    fig_scene = viz_arr.build_figure(
        array,
        rotation,
        pattern,
        fov_range_m=_PAIR_Z,
        fov_stops=_DEFAULT_FOV_STOPS,
        plane_pair=(p1, p2),
    )
    fig_scene.canvas.manager.set_window_title(
        "SAR 阵列与散射点场景（3D，鼠标拖动旋转）"
    )
    fig_scene.suptitle(
        f"2TX×4RX · {rotation.n_stops} 停 · ∠AOB={angle_deg:.2f}°",
        fontsize=11,
        y=0.98,
    )

    zviz = _load_viz_z_plane()
    zviz.plot_z_plane(image, x_axis, y_axis, z_m=BP_Z_M)
    _annotate_bp_figure(plt.gca(), p1, p2, angle_deg)
    fig_bp = plt.gcf()
    fig_bp.canvas.manager.set_window_title(
        f"z={BP_Z_M} m 平面后向投影（步长 {BP_STEP_M} m）"
    )

    print("4) 显示图形：窗口 1 = 3D 场景，窗口 2 = BP 成像（关闭全部窗口后程序结束）")
    plt.show()


def _annotate_bp_figure(ax, p1: np.ndarray, p2: np.ndarray, angle_deg: float) -> None:
    """在 plot_z_plane 生成的图上叠加散射点与说明。"""
    for px, py, tag, color in (
        (p1[0], p1[1], "A", "#7fdbff"),
        (p2[0], p2[1], "B", "#ffdc7f"),
    ):
        ax.scatter(
            [px],
            [py],
            c=color,
            s=28,
            marker="x",
            linewidths=0.9,
            alpha=0.55,
            label=f"散射点 {tag}",
            zorder=5,
        )
    ax.set_title(
        f"SAR 后向投影 — z = {BP_Z_M:.1f} m（5° 双点，{BP_GRID_N}×{BP_GRID_N}）"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.text(
        0.02,
        0.02,
        f"∠AOB={angle_deg:.2f}°\n"
        f"A=({p1[0]:.3f},{p1[1]:.3f})\n"
        f"B=({p2[0]:.3f},{p2[1]:.3f})",
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.45),
    )


if __name__ == "__main__":
    main()
