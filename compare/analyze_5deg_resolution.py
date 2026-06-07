#!/usr/bin/env python3
"""
批量仿真：600 停 SAR 对 5° 世界夹角双点的分辨能力（定量指标）。

在 sar_sim 仓库根目录运行::

  python compare/analyze_5deg_resolution.py
  python compare/analyze_5deg_resolution.py --trials 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _repo_parent = Path(__file__).resolve().parents[2]
    if str(_repo_parent) not in sys.path:
        sys.path.insert(0, str(_repo_parent))

import importlib.util

import numpy as np


def _load_simulate_compare():
    path = Path(__file__).resolve().parent / "simulate_compare.py"
    spec = importlib.util.spec_from_file_location("simulate_compare_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sc = _load_simulate_compare()
BP_STEP_M = _sc.BP_STEP_M
BP_Z_M = _sc.BP_Z_M
_PAIR_ANGLE_DEG = _sc._PAIR_ANGLE_DEG
_PAIR_XY_HALF = _sc._PAIR_XY_HALF
_PAIR_Z = _sc._PAIR_Z
_load_viz_array = _sc._load_viz_array
backproject_z_plane = _sc.backproject_z_plane
bp_grid_axes = _sc.bp_grid_axes
simulate_raw_cube = _sc.simulate_raw_cube
mesh_from_plane_points = _sc.mesh_from_plane_points


def world_angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    na, nb = np.linalg.norm(p1), np.linalg.norm(p2)
    c = float(np.clip(np.dot(p1, p2) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def local_peak_xy(
    mag: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    gt: np.ndarray,
    *,
    search_radius_m: float = 0.025,
) -> tuple[float, float, float]:
    """在 GT 附近窗口内取 BP 峰值位置与幅度。"""
    r_cells = max(1, int(round(search_radius_m / BP_STEP_M)))
    ic = int(np.argmin(np.abs(x_axis - gt[0])))
    ir = int(np.argmin(np.abs(y_axis - gt[1])))
    r0, r1 = max(0, ir - r_cells), min(mag.shape[0], ir + r_cells + 1)
    c0, c1 = max(0, ic - r_cells), min(mag.shape[1], ic + r_cells + 1)
    sub = mag[r0:r1, c0:c1]
    flat_i = int(np.argmax(sub))
    lr, lc = divmod(flat_i, sub.shape[1])
    gr, gc = r0 + lr, c0 + lc
    return float(x_axis[gc]), float(y_axis[gr]), float(mag[gr, gc])


def saddle_ratio(mag: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """沿 AB 连线在像上采样，谷底 / 两峰较大者。"""
    n = 80
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    vals = []
    for x, y in zip(xs, ys):
        ic = int(np.clip(np.argmin(np.abs(x_axis - x)), 0, len(x_axis) - 1))
        ir = int(np.clip(np.argmin(np.abs(y_axis - y)), 0, len(y_axis) - 1))
        vals.append(mag[ir, ic])
    vals = np.asarray(vals, dtype=np.float64)
    peak = float(vals.max())
    if peak <= 0:
        return 1.0
    # 去掉两端各 10% 再取最小（避免落在峰尖上）
    lo, hi = int(0.1 * n), int(0.9 * n)
    mid = vals[lo:hi] if hi > lo else vals
    return float(mid.min()) / peak


def run_trial(seed: int) -> dict:
    viz = _load_viz_array()
    rng = np.random.default_rng(seed)
    p1, p2 = viz.sample_plane_point_pair(
        z=_PAIR_Z,
        xy_half=_PAIR_XY_HALF,
        separation_deg=_PAIR_ANGLE_DEG,
        rng=rng,
    )
    ang = world_angle_deg(p1, p2)
    xy_sep_m = float(np.linalg.norm(p1[:2] - p2[:2]))

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        raw = simulate_raw_cube(mesh_from_plane_points(p1, p2))
        image = backproject_z_plane(raw)
    mag = np.abs(image)
    x_ax, y_ax = bp_grid_axes()

    px1, py1, m1 = local_peak_xy(mag, x_ax, y_ax, p1)
    px2, py2, m2 = local_peak_xy(mag, x_ax, y_ax, p2)
    err1_cm = 100.0 * math.hypot(px1 - p1[0], py1 - p1[1])
    err2_cm = 100.0 * math.hypot(px2 - p2[0], py2 - p2[1])
    peak_sep_cm = 100.0 * math.hypot(px1 - px2, py1 - py2)
    saddle = saddle_ratio(mag, x_ax, y_ax, p1, p2)
    peak_ratio = min(m1, m2) / max(m1, m2) if max(m1, m2) > 0 else 0.0

    # 分辨判据：两峰定位误差 < 1.5 格(7.5mm@5mm步长)，谷底 < 0.65×峰，弱峰/强峰 > 0.25
    cell_cm = BP_STEP_M * 100.0
    resolved = (
        err1_cm <= 1.5 * cell_cm
        and err2_cm <= 1.5 * cell_cm
        and saddle < 0.65
        and peak_ratio > 0.25
    )
    fail_reasons = []
    if err1_cm > 1.5 * cell_cm:
        fail_reasons.append(f"A峰位误差>{1.5*cell_cm:.2f}cm")
    if err2_cm > 1.5 * cell_cm:
        fail_reasons.append(f"B峰位误差>{1.5*cell_cm:.2f}cm")
    if saddle >= 0.65:
        fail_reasons.append("谷底比>=0.65")
    if peak_ratio <= 0.25:
        fail_reasons.append("弱峰过低")

    return {
        "seed": seed,
        "angle_deg": ang,
        "xy_sep_cm": xy_sep_m * 100.0,
        "p1": p1,
        "p2": p2,
        "peak_a_cm": (err1_cm, m1),
        "peak_b_cm": (err2_cm, m2),
        "peak_sep_cm": peak_sep_cm,
        "saddle": saddle,
        "peak_ratio": peak_ratio,
        "resolved": resolved,
        "fail_reasons": fail_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5° 双点分辨能力批量仿真")
    parser.add_argument("--trials", type=int, default=8, help="随机种子个数")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    print("=== 600 停 SAR · 5° 世界夹角双点 · 仿真分辨分析 ===")
    print(f"成像: z={BP_Z_M} m, x/y∈[{-_PAIR_XY_HALF},{_PAIR_XY_HALF}] m, 目标夹角={_PAIR_ANGLE_DEG}°")
    print(f"BP 网格: [{bp_grid_axes()[0][0]:.3f},{bp_grid_axes()[0][-1]:.3f}] m, 步长 {BP_STEP_M} m")
    print(
        "判据: 各峰距 GT < 1.5 格, 连线谷底/峰 < 0.65, 弱峰/强峰 > 0.25\n"
    )

    results = []
    for i in range(args.trials):
        seed = args.seed_start + i
        print(f"--- trial seed={seed} ---")
        r = run_trial(seed)
        results.append(r)
        print(
            f"  ∠AOB={r['angle_deg']:.3f}°  平面间距={r['xy_sep_cm']:.2f} cm  "
            f"分辨={'是' if r['resolved'] else '否'}"
        )
        print(
            f"  A: GT({r['p1'][0]:+.4f},{r['p1'][1]:+.4f}) → 峰误差 {r['peak_a_cm'][0]:.2f} cm"
        )
        print(
            f"  B: GT({r['p2'][0]:+.4f},{r['p2'][1]:+.4f}) → 峰误差 {r['peak_b_cm'][0]:.2f} cm"
        )
        print(
            f"  峰间距={r['peak_sep_cm']:.2f} cm  谷底比={r['saddle']:.3f}  "
            f"弱/强峰={r['peak_ratio']:.3f}"
        )
        if r["fail_reasons"]:
            print(f"  未过严格判据: {', '.join(r['fail_reasons'])}")

    n_ok = sum(1 for r in results if r["resolved"])
    angles = [r["angle_deg"] for r in results]
    saddles = [r["saddle"] for r in results]
    errs = [r["peak_a_cm"][0] for r in results] + [r["peak_b_cm"][0] for r in results]

    print("\n=== 汇总（仿真数据）===")
    print(f"试验次数: {len(results)}  判为可分辨: {n_ok}/{len(results)}")
    print(f"世界夹角 ∠AOB: {min(angles):.3f}° ~ {max(angles):.3f}° (目标 { _PAIR_ANGLE_DEG }°)")
    print(f"峰定位误差: 平均 {np.mean(errs):.2f} cm, 最大 {np.max(errs):.2f} cm")
    print(f"谷底比(越小越像双峰): 平均 {np.mean(saddles):.3f}, 最大 {np.max(saddles):.3f}")

    lam_mm = 3.89
    r_m = _PAIR_Z
    # 全周转台在 R 处横向孔径粗估 ~ 2πR，世界角分辨 ~ λ/(2πR)
    theta_est_deg = math.degrees(lam_mm * 1e-3 / (2 * math.pi * r_m))
    print(
        f"\n参考: lambda~{lam_mm} mm, R~{r_m} m, 全周合成横向角分辨量级 ~ {theta_est_deg:.3f} deg "
        f"(远小于目标 {_PAIR_ANGLE_DEG} deg)"
    )
    n_saddle = sum(1 for r in results if r["saddle"] < 0.65)
    print(f"双峰特征(谷底比<0.65): {n_saddle}/{len(results)}")
    if n_ok == len(results):
        print(f"\n结论: 本批 {len(results)} 组仿真均满足严格判据，600 停 SAR + BP 可分辨 5 deg 双点。")
    elif n_saddle == len(results):
        print(
            f"\n结论: 本批 {len(results)} 组均呈双峰(谷底比<0.65)；"
            f"严格峰位判据通过 {n_ok}/{len(results)} 组。"
            " 600 停 SAR 对世界系 5 deg 双点具备分辨能力，峰位误差偶发受网格/旁瓣影响。"
        )
    else:
        print(f"\n结论: {len(results) - n_saddle} 组谷底比偏高，建议增大 BP 网格密度或检查该 seed。")


if __name__ == "__main__":
    main()
