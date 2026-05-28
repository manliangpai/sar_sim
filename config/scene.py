"""
仿真场景：散射体几何、场景注册与构建。

所有环境定义均在本模块；``simulate.py`` 仅调用 ``build_scene`` / ``scene_from_cli``。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 散射体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointTarget:
    """点散射体（角反或墙面采样点）。"""

    x: float
    y: float
    z: float
    amplitude: float = 1.0


def targets_to_xyz(targets: List[PointTarget]) -> np.ndarray:
    """(N, 3) 目标坐标。"""
    return np.array([[t.x, t.y, t.z] for t in targets], dtype=np.float64)


# ---------------------------------------------------------------------------
# 场景 1：两角反
# ---------------------------------------------------------------------------


def two_corner_reflector_scene() -> List[PointTarget]:
    """
    两个角反。

    - (0, 0, 2) m
    - (1, 0, 2) m
    """
    return [
        PointTarget(x=0.0, y=0.0, z=2.0, amplitude=1.0),
        PointTarget(x=1.0, y=0.0, z=2.0, amplitude=1.0),
    ]


# ---------------------------------------------------------------------------
# 场景 2：凹形五面墙室
# ---------------------------------------------------------------------------
#
# 俯视图 (开口朝向原点，雷达主瓣 +Z):
#
#        y=+1  +---------------------------+  y=+1
#              |          墙2 (z=3)        |
#        y=-1  +---------------------------+  y=-1
#     墙1      |    开口 (朝向雷达)         |      墙3
#   x:-1.2~-0.2|   x:-0.2 ~ 0.2, z:1~3    | x:0.2~1.2
#   z=1        |                           | z=1
#              侧墙L x=-0.2        侧墙R x=0.2


@dataclass(frozen=True)
class WallRect:
    """轴对齐矩形墙面。"""

    name: str
    fixed_axis: str
    fixed_value: float
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    varying_axes: Tuple[str, str] = ("x", "y")


def _concave_room_walls() -> List[WallRect]:
    return [
        WallRect("wall1_left_front", "z", 1.0, -1.2, -0.2, -1.0, 1.0, ("x", "y")),
        WallRect("wall2_back", "z", 3.0, -0.2, 0.2, -1.0, 1.0, ("x", "y")),
        WallRect("wall3_right_front", "z", 1.0, 0.2, 1.2, -1.0, 1.0, ("x", "y")),
        WallRect("side_L", "x", -0.2, -1.0, 1.0, 1.0, 3.0, ("y", "z")),
        WallRect("side_R", "x", 0.2, -1.0, 1.0, 1.0, 3.0, ("y", "z")),
    ]


def _axis_vals(v0: float, v1: float, spacing: float) -> np.ndarray:
    lo, hi = (v0, v1) if v0 <= v1 else (v1, v0)
    if hi <= lo:
        return np.array([(lo + hi) * 0.5], dtype=np.float64)
    n = max(int(np.floor((hi - lo) / spacing)) + 1, 2)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _sample_wall_rect(
    wall: WallRect,
    spacing_m: float,
) -> List[PointTarget]:
    a_vals = _axis_vals(wall.a_min, wall.a_max, spacing_m)
    b_vals = _axis_vals(wall.b_min, wall.b_max, spacing_m)
    ax0, ax1 = wall.varying_axes
    targets: List[PointTarget] = []
    for a in a_vals:
        for b in b_vals:
            coords = {ax0: float(a), ax1: float(b), wall.fixed_axis: wall.fixed_value}
            targets.append(
                PointTarget(
                    x=coords["x"],
                    y=coords["y"],
                    z=coords["z"],
                    amplitude=1.0,
                )
            )
    return targets


def concave_room_scene(
    spacing_m: float = 0.05,
    wall_amplitude: float | None = None,
    include_two_corner_reflectors: bool = False,
    corner_amplitude: float = 1.0,
) -> List[PointTarget]:
    """
    凹形墙室：五面墙均匀采样为点散射体。

    Parameters
    ----------
    spacing_m
        墙面采样间距 (m)。
    wall_amplitude
        每点幅度；默认按点数缩放，避免相干叠加过强。
    include_two_corner_reflectors
        是否额外加入与 ``two_corners`` 场景相同的两角反 (0,0,2)、(1,0,2)。
    """
    per_wall = [_sample_wall_rect(w, spacing_m) for w in _concave_room_walls()]
    n_wall_pts = sum(len(p) for p in per_wall)
    amp = (
        wall_amplitude
        if wall_amplitude is not None
        else 0.15 / max(np.sqrt(n_wall_pts / 100.0), 1.0)
    )

    targets: List[PointTarget] = [
        PointTarget(p.x, p.y, p.z, amplitude=amp) for pts in per_wall for p in pts
    ]
    if include_two_corner_reflectors:
        targets.extend(
            [
                PointTarget(0.0, 0.0, 2.0, amplitude=corner_amplitude),
                PointTarget(1.0, 0.0, 2.0, amplitude=corner_amplitude),
            ]
        )
    return targets


def _concave_room_summary(targets: Sequence[PointTarget]) -> str:
    lines = [
        "concave room (5 walls, opening toward origin +Z):",
        f"  scatterers: {len(targets)}",
        f"  walls: {[w.name for w in _concave_room_walls()]}",
    ]
    if targets:
        xyz = np.array([[t.x, t.y, t.z] for t in targets])
        lines.append(
            f"  bbox x[{xyz[:, 0].min():.2f},{xyz[:, 0].max():.2f}] "
            f"y[{xyz[:, 1].min():.2f},{xyz[:, 1].max():.2f}] "
            f"z[{xyz[:, 2].min():.2f},{xyz[:, 2].max():.2f}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 场景注册
# ---------------------------------------------------------------------------

SceneBuilder = Callable[..., List[PointTarget]]


@dataclass(frozen=True)
class SceneEntry:
    """注册表项：构建函数、默认输出文件名、简述。"""

    name: str
    description: str
    output_filename: str
    build: SceneBuilder
    summarize: Callable[[Sequence[PointTarget]], str] | None = None


DEFAULT_SCENE_NAME = "two_corners"

SCENES: Dict[str, SceneEntry] = {
    "two_corners": SceneEntry(
        name="two_corners",
        description="2 corner reflectors at (0,0,2) and (1,0,2)",
        output_filename="ti1843_2t4r_two_corner_reflectors.npz",
        build=two_corner_reflector_scene,
    ),
    "concave_room": SceneEntry(
        name="concave_room",
        description="concave 5-wall room (point-sampled walls)",
        output_filename="ti1843_2t4r_concave_room.npz",
        build=concave_room_scene,
        summarize=_concave_room_summary,
    ),
}

def scene_names() -> List[str]:
    """已注册场景名。"""
    return list(SCENES.keys())


def resolve_scene_name(name: str) -> str:
    if name not in SCENES:
        raise ValueError(f"未知场景 {name!r}，可选: {', '.join(scene_names())}")
    return name


def get_scene_entry(name: str) -> SceneEntry:
    return SCENES[resolve_scene_name(name)]


def build_scene(name: str = DEFAULT_SCENE_NAME, **kwargs) -> List[PointTarget]:
    """按注册名构建散射体列表。"""
    return get_scene_entry(name).build(**kwargs)


def scene_output_filename(name: str) -> str:
    return get_scene_entry(name).output_filename


def describe_scene(name: str, targets: Sequence[PointTarget]) -> str:
    """场景一行描述 + 可选详细 summary。"""
    entry = get_scene_entry(name)
    lines = [entry.description, f"  scatterers: {len(targets)}"]
    if entry.summarize is not None:
        lines.append(entry.summarize(targets))
    return "\n".join(lines)


def add_scene_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """向 ArgumentParser 添加场景相关参数（供 simulate 调用）。"""
    parser.add_argument(
        "--scene",
        choices=tuple(SCENES.keys()),
        default=DEFAULT_SCENE_NAME,
        help="场景名：two_corners | concave_room",
    )
    parser.add_argument(
        "--wall-spacing",
        type=float,
        default=0.05,
        metavar="M",
        help="[concave_room] 墙面采样间距 (m)",
    )
    parser.add_argument(
        "--add-two-corner-reflectors",
        action="store_true",
        help="[concave_room] 额外加入 two_corners 场景中的两角反",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 npz 文件名（默认取场景注册名）",
    )


def scene_from_cli(args: argparse.Namespace) -> tuple[str, List[PointTarget], str]:
    """
    从 CLI 命名空间构建场景。

    Returns
    -------
    scene_name, targets, output_filename
    """
    name = resolve_scene_name(args.scene)
    entry = get_scene_entry(name)
    if name == "concave_room":
        targets = entry.build(
            spacing_m=args.wall_spacing,
            include_two_corner_reflectors=args.add_two_corner_reflectors,
        )
    else:
        targets = entry.build()
    out = args.output if args.output is not None else entry.output_filename
    return name, targets, out
