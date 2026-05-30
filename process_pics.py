#!/usr/bin/env python3
"""
批量处理 pixel_pattern 场景：距离 FFT + z=0.6 m 平面双基地后向投影。

对每个数字/字母图案读取对应 raw SAR 立方体，
在 x/y ∈ [-10, +10] cm、格距 1 cm 的 20×20 网格上成像，保存 (20, 20) 复数图。

运行（在 sar_sim 仓库根目录）::

  python process_pics.py
  python process_pics.py --pattern 0 A B
  python process_pics.py --simulate-missing

输入（默认）: output/raw_radar_data_z0.6/{NAME}.npz
输出（默认）: output/processed_radar_data_z0.6/{NAME}.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np

from sar_sim.config import ArrayConfig, RadarConfig, SarRotationConfig
from sar_sim.config.scene import (
    DEFAULT_CUBE_SUBDIV,
    DEFAULT_PATTERN_PICS_DIR,
    PIXEL_CELL_M,
    PIXEL_GRID_N,
    PIXEL_PLANE_SIZE_M,
    PIXEL_PLANE_Z_M,
    _pattern_sort_key,
    pixel_pattern_names,
    pixel_pattern_scene,
)
from sar_sim.simulate_pics import load_sar_cube, save_raw_cube, simulate_sar_rotation_cube
from sar_sim.visualize_on_a_z_plane import backproject_z_plane, range_fft_cube

_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = _PACKAGE_ROOT / "output" / "raw_radar_data_z0.6"
DEFAULT_OUTPUT_DIR = _PACKAGE_ROOT / "output" / "processed_radar_data_z0.6"
DEFAULT_PICS_DIR = DEFAULT_PATTERN_PICS_DIR


def pixel_imaging_axes(
    n: int = PIXEL_GRID_N,
    cell_m: float = PIXEL_CELL_M,
) -> tuple[np.ndarray, np.ndarray]:
    """20×20 格心坐标，x/y ∈ [-10, +10] cm（±half + 半格）。"""
    half = n * cell_m * 0.5
    axis = (-half + (np.arange(n, dtype=np.float64) + 0.5) * cell_m).astype(
        np.float64
    )
    return axis, axis.copy()


def raw_npz_path(pattern_name: str, input_dir: Path) -> Path:
    return input_dir / f"{pattern_name}.npz"


def process_one_pattern(
    pattern_name: str,
    *,
    input_dir: Path,
    output_dir: Path,
    pics_dir: Path | None = None,
    z_m: float = PIXEL_PLANE_Z_M,
    simulate_if_missing: bool = False,
) -> tuple[np.ndarray, Path]:
    """读取 raw → 距离 FFT → z 平面 BP，返回 (20,20) 图像与保存路径。"""
    in_path = raw_npz_path(pattern_name, input_dir)
    if not in_path.is_file():
        if not simulate_if_missing:
            raise FileNotFoundError(
                f"缺少仿真数据 {in_path}，请先将 raw 放入 {input_dir}，"
                f"或加 --simulate-missing 自动生成"
            )
        print(f"  simulate {pattern_name} → {input_dir} ...", flush=True)
        targets = pixel_pattern_scene(pattern_name, pics_dir=pics_dir)
        cube = simulate_sar_rotation_cube(targets=targets)
        in_path = save_raw_cube(cube, in_path.name, output_dir=input_dir)

    raw, start_angle_rad = load_sar_cube(in_path)
    rotation = SarRotationConfig(
        n_stops=raw.shape[0], start_angle_rad=start_angle_rad
    )
    range_cube = range_fft_cube(raw)

    x_axis, y_axis = pixel_imaging_axes()
    if len(x_axis) != PIXEL_GRID_N or len(y_axis) != PIXEL_GRID_N:
        raise RuntimeError(
            f"成像轴长须为 {PIXEL_GRID_N}，得到 {len(x_axis)}×{len(y_axis)}"
        )

    radar = RadarConfig()
    array = ArrayConfig()
    image, x_axis, y_axis = backproject_z_plane(
        range_cube,
        z_m=z_m,
        x_axis=x_axis,
        y_axis=y_axis,
        radar=radar,
        array=array,
        rotation=rotation,
    )
    if image.shape != (PIXEL_GRID_N, PIXEL_GRID_N):
        raise RuntimeError(f"图像形状须 ({PIXEL_GRID_N},{PIXEL_GRID_N})，得到 {image.shape}")

    out_path = output_dir / f"{pattern_name}.npz"
    np.savez_compressed(
        out_path,
        image=image.astype(np.complex64, copy=False),
        magnitude=np.abs(image).astype(np.float32),
        x_m=x_axis,
        y_m=y_axis,
        z_m=np.float64(z_m),
        pattern_name=np.array(pattern_name),
        raw_npz=np.array(in_path.name),
        start_angle_rad=np.float64(start_angle_rad),
    )
    return image, out_path


def process_all(
    pattern_names: list[str],
    *,
    input_dir: Path,
    output_dir: Path,
    pics_dir: Path | None = None,
    z_m: float = PIXEL_PLANE_Z_M,
    simulate_if_missing: bool = False,
) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}

    for name in pattern_names:
        t0 = time.perf_counter()
        print(f"processing {name} ...", flush=True)
        image, out_path = process_one_pattern(
            name,
            input_dir=input_dir,
            output_dir=output_dir,
            pics_dir=pics_dir,
            z_m=z_m,
            simulate_if_missing=simulate_if_missing,
        )
        elapsed = time.perf_counter() - t0
        peak = float(np.abs(image).max())
        entry = {
            "file": out_path.name,
            "shape": list(image.shape),
            "peak_magnitude": peak,
            "elapsed_s": round(elapsed, 2),
        }
        results[name] = entry
        print(f"  saved {out_path}  peak={peak:.4g}  ({elapsed:.1f}s)", flush=True)

    for path in output_dir.glob("*.json"):
        path.unlink()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pixel_pattern raw → z=0.6 m 平面 20×20 后向投影"
    )
    parser.add_argument(
        "--pattern",
        nargs="*",
        default=None,
        metavar="NAME",
        help="图案名（默认 output/pics 下全部）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"仿真 raw 目录（默认 {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"处理后输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--pics-dir",
        type=Path,
        default=DEFAULT_PICS_DIR,
        metavar="DIR",
        help=f"GT npy 目录（默认 {DEFAULT_PICS_DIR}）",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=PIXEL_PLANE_Z_M,
        metavar="M",
        help=f"成像 z 平面 (m)，默认 {PIXEL_PLANE_Z_M}",
    )
    parser.add_argument(
        "--simulate-missing",
        action="store_true",
        help="若 input-dir 缺少 {NAME}.npz，先仿真并写入 input-dir",
    )
    args = parser.parse_args()

    names = args.pattern if args.pattern else pixel_pattern_names(args.pics_dir)
    if not names:
        raise SystemExit("未找到图案：请先运行 test_code/generate_pattern_pics.py")

    print(f"patterns: {len(names)}")
    print(f"  input : {args.input_dir.resolve()}")
    print(f"  output: {args.output_dir.resolve()}")
    if args.pics_dir:
        print(f"  pics  : {args.pics_dir.resolve()}")
    print(
        f"  grid  : {PIXEL_GRID_N}×{PIXEL_GRID_N}, "
        f"±{PIXEL_PLANE_SIZE_M*0.5*1e2:.0f} cm, step {PIXEL_CELL_M*1e2:.0f} cm, "
        f"z={args.z} m"
    )

    process_all(
        names,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pics_dir=args.pics_dir,
        z_m=args.z,
        simulate_if_missing=args.simulate_missing,
    )
    print(f"done: {len(names)} files → {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
