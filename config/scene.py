"""
场景坐标与 PLY mesh 加载。

世界坐标系（传送带扫描工位）：
  Z = 0    雷达阵列所在平面（阵列绕 +Z 转台旋转，停点 1 时 PCB 在 y=z=0）
  Z = 0.5  传送带顶面；阵列到传送带垂直距离 0.5 m
  被扫物体放在传送带上，例如金属板顶面 z=0.5 m、厚度 5 cm 则 z∈[0.45, 0.5]

默认模型 data/metal_plate_20x20x5cm.ply（20 cm × 20 cm × 5 cm 长方体）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLY_PATH = _PACKAGE_ROOT / "data" / "metal_plate_20x20x5cm.ply"


def raw_npz_stem(ply_path: Path | str | None = None) -> str:
    """仿真 raw npz 基名：与 PLY 文件名一致（不含扩展名）。"""
    return Path(ply_path or DEFAULT_PLY_PATH).stem

RADAR_PLANE_Z_M = 0.0
CONVEYOR_Z_M = 0.5

PLATE_X_HALF_M = 0.1
PLATE_Y_HALF_M = 0.1
PLATE_Z_MIN_M = 0.45
PLATE_Z_MAX_M = CONVEYOR_Z_M
PLATE_Z_CENTER_M = 0.5 * (PLATE_Z_MIN_M + PLATE_Z_MAX_M)


def radar_output_dir(prefix: str) -> Path:
    return _PACKAGE_ROOT / "output" / prefix


class ScatterMode(str, Enum):
    VERTEX = "vertex"
    FACE_CENTER = "face_center"
    FACE_VISIBLE = "face_visible"


@dataclass(frozen=True)
class ReflectionConfig:
    enabled: bool = False
    points: tuple[tuple[float, float, float], ...] = ()
    coefficients: tuple[float, ...] = ()

    @classmethod
    def conveyor_belt_grid(
        cls,
        *,
        z: float = CONVEYOR_Z_M,
        x_range: tuple[float, float] = (-0.15, 0.15),
        y_range: tuple[float, float] = (-0.15, 0.15),
        n: int = 3,
        coefficient: float = 0.35,
    ) -> "ReflectionConfig":
        """传送带平面上的反射采样点（TX→物体→传送带→RX 多径）。"""
        xs = np.linspace(x_range[0], x_range[1], n)
        ys = np.linspace(y_range[0], y_range[1], n)
        pts = [(float(x), float(y), float(z)) for x in xs for y in ys]
        return cls(enabled=True, points=tuple(pts), coefficients=tuple([coefficient] * len(pts)))


@dataclass(frozen=True)
class ScatterMesh:
    positions: np.ndarray
    area_weight: np.ndarray
    normals: np.ndarray | None
    mode: ScatterMode
    ply_path: str = ""

    @property
    def n_scatterers(self) -> int:
        return int(self.positions.shape[0])


def _parse_ply_header(data: bytes) -> tuple[bool, int, int, int]:
    text = data[: min(len(data), 65536)].split(b"\n")
    is_binary = False
    n_vert = n_face = 0
    header_end = 0
    for i, raw in enumerate(text):
        line = raw.decode("ascii", errors="ignore").strip()
        if line == "end_header":
            header_end = sum(len(text[j]) + 1 for j in range(i + 1))
            break
        if line.startswith("format binary"):
            is_binary = True
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_vert = int(parts[2])
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
            n_face = int(parts[2])
    if header_end == 0 or n_vert <= 0:
        raise ValueError("PLY 头解析失败")
    return is_binary, header_end, n_vert, n_face


def _load_ply_binary(data: bytes, header_end: int, n_vert: int, n_face: int) -> tuple[np.ndarray, np.ndarray]:
    off = header_end
    vertices = np.frombuffer(data, dtype="<f4", count=n_vert * 3, offset=off).reshape(n_vert, 3).astype(
        np.float64
    )
    off += n_vert * 12
    faces = np.zeros((n_face, 3), dtype=np.int32)
    for fi in range(n_face):
        n = data[off]
        if n != 3:
            raise ValueError(f"仅支持三角面，面 {fi} 有 {n} 顶点")
        i0, i1, i2 = struct.unpack_from("<iii", data, off + 1)
        faces[fi] = (i0, i1, i2)
        off += 13
    return vertices, faces


def _load_ply_ascii_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    lines = text.splitlines()
    is_bin, _, n_vert, n_face = _parse_ply_header(text.encode("utf-8"))
    if is_bin:
        raise ValueError("请用 load_ply() 读取 binary PLY")
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
    vert_lines = lines[header_end : header_end + n_vert]
    vertices = np.array([[float(x) for x in ln.split()] for ln in vert_lines], dtype=np.float64)
    faces: list[list[int]] = []
    for line in lines[header_end + n_vert : header_end + n_vert + n_face]:
        parts = line.split()
        if int(parts[0]) != 3:
            raise ValueError(f"仅支持三角面: {line}")
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return vertices, np.asarray(faces, dtype=np.int32)


def load_ply(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """读取 ASCII 或 binary PLY。"""
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(b"ply"):
        raise ValueError(f"不是 PLY: {path}")
    is_binary, header_end, n_vert, n_face = _parse_ply_header(data)
    if is_binary:
        return _load_ply_binary(data, header_end, n_vert, n_face)
    return _load_ply_ascii_text(data.decode("utf-8"))


def load_ply_ascii(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """兼容旧名；支持 ASCII 与 binary。"""
    return load_ply(path)


def _triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0)))


def _triangle_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    v1 = tri[:, 1] - tri[:, 0]
    v2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(v1, v2)
    denom = np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    return (cross / denom).astype(np.float64)


def _triangle_centers_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centers = tri.mean(axis=1)
    areas = np.array(
        [_triangle_area(tri[i, 0], tri[i, 1], tri[i, 2]) for i in range(len(faces))],
        dtype=np.float64,
    )
    return centers, areas


def _subdivide_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, areas = _triangle_centers_areas(vertices, faces)
    normals = _triangle_normals(vertices, faces)
    if levels == 0:
        return centers, areas, normals

    new_centers: list[np.ndarray] = []
    new_areas: list[float] = []
    new_normals: list[np.ndarray] = []
    for fi in range(faces.shape[0]):
        v0, v1, v2 = vertices[faces[fi, 0]], vertices[faces[fi, 1]], vertices[faces[fi, 2]]
        n0 = normals[fi]
        stack: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [(v0, v1, v2)]
        for _ in range(levels):
            nxt: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for a, b, c in stack:
                m_ab, m_bc, m_ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
                nxt.extend([(a, m_ab, m_ca), (m_ab, b, m_bc), (m_ca, m_bc, c), (m_ab, m_bc, m_ca)])
            stack = nxt
        for a, b, c in stack:
            new_centers.append((a + b + c) / 3.0)
            new_areas.append(_triangle_area(a, b, c))
            new_normals.append(n0)
    return (
        np.asarray(new_centers, dtype=np.float64),
        np.asarray(new_areas, dtype=np.float64),
        np.asarray(new_normals, dtype=np.float64),
    )


def build_scatter_mesh(
    ply_path: Path | str,
    *,
    mode: ScatterMode = ScatterMode.FACE_VISIBLE,
    face_subdiv: int = 1,
    total_amplitude: float = 1.0,
) -> ScatterMesh:
    path = Path(ply_path)
    vertices, faces = load_ply(path)

    if mode == ScatterMode.VERTEX:
        positions = vertices.copy()
        areas = np.ones(len(vertices), dtype=np.float64)
        normals = None
    else:
        positions, areas, normals = _subdivide_triangles(vertices, faces, levels=face_subdiv)

    area_sum = float(areas.sum())
    if area_sum <= 0:
        raise ValueError("散射体面积为 0")
    area_weight = (total_amplitude * areas / area_sum).astype(np.float64)

    return ScatterMesh(
        positions=positions.astype(np.float64),
        area_weight=area_weight,
        normals=normals,
        mode=mode,
        ply_path=str(path.resolve()),
    )


def scatter_mesh_from_points(
    positions: np.ndarray,
    *,
    amplitudes: np.ndarray | None = None,
) -> ScatterMesh:
    """任意坐标点散射体（compare 双点实验等）。"""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions 须为 (N, 3)，得到 {pos.shape}")
    if amplitudes is None:
        amp = np.ones(pos.shape[0], dtype=np.float64)
    else:
        amp = np.asarray(amplitudes, dtype=np.float64)
        if amp.shape != (pos.shape[0],):
            raise ValueError(f"amplitudes 须为 ({pos.shape[0]},)，得到 {amp.shape}")
    return ScatterMesh(
        positions=pos,
        area_weight=amp,
        normals=None,
        mode=ScatterMode.VERTEX,
        ply_path="",
    )
