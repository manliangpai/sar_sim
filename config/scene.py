"""
pixel_pattern 场景：20×20、1 cm 格距金属立方体体素化散射体。

GT 数组位于 output/pics/{NAME}.npy（数字/字母、Fashion-MNIST 等训练图案）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN_PICS_DIR = _PACKAGE_ROOT / "output" / "pics"


@dataclass(frozen=True)
class PointTarget:
    """点散射体。"""

    x: float
    y: float
    z: float
    amplitude: float = 1.0


def targets_to_xyz(targets: List[PointTarget]) -> np.ndarray:
    """(N, 3) 目标坐标。"""
    return np.array([[t.x, t.y, t.z] for t in targets], dtype=np.float64)


# ---------------------------------------------------------------------------
# Z=0.6 m 平面 20×20 像素图案（1 cm³ 金属立方体体素化）
# ---------------------------------------------------------------------------

PIXEL_GRID_N = 20
PIXEL_CELL_M = 0.01
PIXEL_PLANE_Z_M = 0.6
PIXEL_PLANE_SIZE_M = PIXEL_GRID_N * PIXEL_CELL_M
DEFAULT_CUBE_SUBDIV = 3


def _pattern_sort_key(name: str) -> tuple[str, int] | tuple[str]:
    """
    图案名排序：``{前缀}_{编号}`` 按编号自然数序（1,2,…,10 而非 1,10,2）；
    纯数字名（0-9）按数值；其余按名称。
    """
    if "_" in name:
        prefix, _, suffix = name.rpartition("_")
        if suffix.isdigit():
            return (prefix, int(suffix))
    if name.isdigit():
        return ("", int(name))
    return (name,)


def pixel_pattern_names(pics_dir: Path | None = None) -> List[str]:
    """pics 目录下可用图案名（*.npy）。"""
    root = pics_dir or DEFAULT_PATTERN_PICS_DIR
    if not root.is_dir():
        return []
    return sorted(
        (p.stem for p in root.glob("*.npy")),
        key=_pattern_sort_key,
    )


def load_pixel_pattern_gt(path: Path | str) -> np.ndarray:
    """
    读取 GT 数组。

    返回 (20, 20) uint8，1=有 1 cm³ 金属块，0=空。
    行 0 为显示最上行，与 generate_pattern_pics 一致。
    """
    path = Path(path)
    arr = np.asarray(np.load(path))
    if arr.shape != (PIXEL_GRID_N, PIXEL_GRID_N):
        raise ValueError(f"{path}: 形状须 ({PIXEL_GRID_N},{PIXEL_GRID_N})，得到 {arr.shape}")
    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError(f"{path}: GT 须仅含 0/1")
    return arr.astype(np.uint8, copy=False)


def resolve_pixel_pattern_gt(
    pattern_name: str,
    pics_dir: Path | None = None,
) -> Path:
    """按图案名解析 GT：pics_dir/{name}.npy。"""
    root = pics_dir or DEFAULT_PATTERN_PICS_DIR
    path = root / f"{pattern_name}.npy"
    if path.is_file():
        return path
    available = ", ".join(pixel_pattern_names(root)[:8])
    hint = f" 可用: {available}..." if available else " （pics 目录为空）"
    raise FileNotFoundError(f"未找到 GT {pattern_name}{hint}")


def _pixel_cell_origin_m(col: int, display_row: int) -> Tuple[float, float, float]:
    """格点左下前角 (x0,y0,z0)；立方体 x/y 居中于原点，z 以 PIXEL_PLANE_Z_M 为中心。"""
    if not (0 <= col < PIXEL_GRID_N and 0 <= display_row < PIXEL_GRID_N):
        raise ValueError(
            f"格点索引须在 [0, {PIXEL_GRID_N}) 内，得到 ({col}, {display_row})"
        )
    math_row = PIXEL_GRID_N - 1 - display_row
    half = PIXEL_PLANE_SIZE_M * 0.5
    x0 = -half + col * PIXEL_CELL_M
    y0 = -half + math_row * PIXEL_CELL_M
    z0 = PIXEL_PLANE_Z_M - PIXEL_CELL_M * 0.5
    return x0, y0, z0


def _sample_metal_cube(
    x0: float,
    y0: float,
    z0: float,
    *,
    cell_m: float = PIXEL_CELL_M,
    cube_subdiv: int = DEFAULT_CUBE_SUBDIV,
    block_amplitude: float = 1.0,
) -> List[PointTarget]:
    """将 1 个 cell_m³ 金属块采样为 cube_subdiv³ 个点散射体。"""
    if cube_subdiv < 1:
        raise ValueError(f"cube_subdiv 须 >= 1，得到 {cube_subdiv}")
    step = cell_m / cube_subdiv
    n_pts = cube_subdiv**3
    point_amp = block_amplitude / n_pts
    targets: List[PointTarget] = []
    for ix in range(cube_subdiv):
        for iy in range(cube_subdiv):
            for iz in range(cube_subdiv):
                targets.append(
                    PointTarget(
                        x=x0 + (ix + 0.5) * step,
                        y=y0 + (iy + 0.5) * step,
                        z=z0 + (iz + 0.5) * step,
                        amplitude=point_amp,
                    )
                )
    return targets


def targets_from_pixel_mask(
    mask: Union[np.ndarray, str, Path],
    *,
    cube_subdiv: int = DEFAULT_CUBE_SUBDIV,
    block_amplitude: float = 1.0,
) -> List[PointTarget]:
    """(20, 20) GT 掩膜 → 体素化金属立方体散射体列表。"""
    if isinstance(mask, (str, Path)):
        gt = load_pixel_pattern_gt(mask)
    else:
        gt = np.asarray(mask)
    if gt.shape != (PIXEL_GRID_N, PIXEL_GRID_N):
        raise ValueError(f"掩膜须为 ({PIXEL_GRID_N},{PIXEL_GRID_N})，得到 {gt.shape}")
    if not np.all(np.isin(gt, (0, 1))):
        raise ValueError("GT 须仅含 0/1")
    gt = gt.astype(np.uint8)

    targets: List[PointTarget] = []
    n_blocks = 0
    for display_row in range(PIXEL_GRID_N):
        for col in range(PIXEL_GRID_N):
            if not gt[display_row, col]:
                continue
            n_blocks += 1
            x0, y0, z0 = _pixel_cell_origin_m(col, display_row)
            targets.extend(
                _sample_metal_cube(
                    x0,
                    y0,
                    z0,
                    cube_subdiv=cube_subdiv,
                    block_amplitude=block_amplitude,
                )
            )
    if n_blocks == 0:
        raise ValueError("GT 中没有任何金属格点（值为 1）")
    return targets


def pixel_pattern_scene(
    pattern_name: str = "0",
    *,
    pics_dir: Path | None = None,
    cube_subdiv: int = DEFAULT_CUBE_SUBDIV,
    block_amplitude: float = 1.0,
) -> List[PointTarget]:
    """
    从 output/pics/{pattern_name}.npy 构建场景。

    每个 1 cm 金属格点划分为 cube_subdiv×cube_subdiv×cube_subdiv 散射点
    （默认 3³=27 点/块）；块内各点幅度 = block_amplitude / 27。
    """
    path = resolve_pixel_pattern_gt(pattern_name, pics_dir=pics_dir)
    mask = load_pixel_pattern_gt(path)
    return targets_from_pixel_mask(
        mask,
        cube_subdiv=cube_subdiv,
        block_amplitude=block_amplitude,
    )


def describe_pixel_pattern(
    pattern_name: str,
    targets: Sequence[PointTarget],
) -> str:
    """pixel_pattern 场景一行描述 + 散射体统计。"""
    lines = [
        f"pixel_pattern '{pattern_name}' @ z={PIXEL_PLANE_Z_M} m",
        (
            f"  grid: {PIXEL_GRID_N}×{PIXEL_GRID_N} @ {PIXEL_CELL_M*1e2:.0f} cm, "
            f"voxelized 1 cm³ cubes"
        ),
        f"  scatterers: {len(targets)}",
        (
            f"  extent: ±{PIXEL_PLANE_SIZE_M*0.5*1e2:.0f} cm in x/y, "
            f"z∈[{PIXEL_PLANE_Z_M - PIXEL_CELL_M*0.5:.3f},"
            f"{PIXEL_PLANE_Z_M + PIXEL_CELL_M*0.5:.3f}] m"
        ),
    ]
    if targets:
        xyz = np.array([[t.x, t.y, t.z] for t in targets], dtype=np.float64)
        lines.append(
            f"  bbox x[{xyz[:, 0].min():.3f},{xyz[:, 0].max():.3f}] "
            f"y[{xyz[:, 1].min():.3f},{xyz[:, 1].max():.3f}] "
            f"z[{xyz[:, 2].min():.3f},{xyz[:, 2].max():.3f}]"
        )
    return "\n".join(lines)
