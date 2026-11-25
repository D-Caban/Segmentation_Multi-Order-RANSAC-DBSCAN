"""Point cloud and Gaussian splat I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

# Graphdeco / 3DGS training PLY property names we preserve when present.
GAUSSIAN_ATTR_PREFIXES = (
    "nx",
    "ny",
    "nz",
    "f_dc_",
    "f_rest_",
    "opacity",
    "scale_",
    "rot_",
)


@dataclass
class CloudData:
    """Loaded cloud: geometry for segmentation plus optional per-point attributes."""

    points: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    is_gaussian: bool = False
    source_path: str | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.float64)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None and len(self.colors) == len(self.points):
            pcd.colors = o3d.utility.Vector3dVector(np.clip(self.colors, 0.0, 1.0))
        if self.normals is not None and len(self.normals) == len(self.points):
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd

    def select(self, indices: np.ndarray | list[int]) -> CloudData:
        idx = np.asarray(indices, dtype=np.int64)
        attrs = {k: v[idx] for k, v in self.attributes.items()}
        colors = self.colors[idx] if self.colors is not None else None
        normals = self.normals[idx] if self.normals is not None else None
        return CloudData(
            points=self.points[idx],
            colors=colors,
            normals=normals,
            attributes=attrs,
            is_gaussian=self.is_gaussian,
            source_path=self.source_path,
        )


def _vertex_names(vertex: Any) -> set[str]:
    return {p.name for p in vertex.properties}


def _is_gaussian_ply(names: set[str]) -> bool:
    needed = {"x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"}
    return needed.issubset(names)


def _rgb_from_gaussian(vertex: Any, names: set[str]) -> np.ndarray | None:
    if not {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        return None
    # Graphdeco SH DC → approximate RGB in [0, 1]
    sh_c0 = 0.28209479177387814
    rgb = np.column_stack(
        [
            0.5 + sh_c0 * np.asarray(vertex["f_dc_0"], dtype=np.float64),
            0.5 + sh_c0 * np.asarray(vertex["f_dc_1"], dtype=np.float64),
            0.5 + sh_c0 * np.asarray(vertex["f_dc_2"], dtype=np.float64),
        ]
    )
    return np.clip(rgb, 0.0, 1.0)


def load_gaussian_ply(path: Path) -> CloudData:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = _vertex_names(vertex)
    points = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ]
    )
    attrs: dict[str, np.ndarray] = {}
    for name in names:
        if name in {"x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"}:
            continue
        if name.startswith(GAUSSIAN_ATTR_PREFIXES) or name in {
            "opacity",
        }:
            attrs[name] = np.asarray(vertex[name])
        elif any(name.startswith(p) for p in GAUSSIAN_ATTR_PREFIXES):
            attrs[name] = np.asarray(vertex[name])

    # Capture any remaining non-xyz numeric properties for round-trip.
    for name in names:
        if name in {"x", "y", "z"} or name in attrs:
            continue
        if name in {"red", "green", "blue", "nx", "ny", "nz", "alpha"}:
            continue
        try:
            attrs[name] = np.asarray(vertex[name])
        except Exception:
            continue

    colors = _rgb_from_gaussian(vertex, names)
    normals = None
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.column_stack(
            [
                np.asarray(vertex["nx"], dtype=np.float64),
                np.asarray(vertex["ny"], dtype=np.float64),
                np.asarray(vertex["nz"], dtype=np.float64),
            ]
        )
    return CloudData(
        points=points,
        colors=colors,
        normals=normals,
        attributes=attrs,
        is_gaussian=True,
        source_path=str(path),
    )


def load_cloud(path: str | Path) -> CloudData:
    """Load PLY/PCD. Gaussian training PLYs keep extra attributes."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".ply":
        try:
            from plyfile import PlyData

            ply = PlyData.read(str(path))
            names = _vertex_names(ply["vertex"])
            if _is_gaussian_ply(names):
                return load_gaussian_ply(path)
        except Exception:
            pass

    if suffix not in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}:
        raise ValueError(f"Unsupported format: {suffix} ({path})")

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty or unreadable point cloud: {path}")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    return CloudData(
        points=points,
        colors=colors if colors is not None and len(colors) else None,
        normals=normals if normals is not None and len(normals) else None,
        is_gaussian=False,
        source_path=str(path),
    )


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    attributes: dict[str, np.ndarray] | None = None,
) -> None:
    """Write XYZ (+ optional RGB and extra scalar/vector attributes) as binary PLY."""
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    dtype_list: list[tuple[str, str]] = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None and len(colors) == n:
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    attr_arrays: dict[str, np.ndarray] = {}
    if attributes:
        for name, arr in attributes.items():
            arr = np.asarray(arr)
            if len(arr) != n:
                continue
            if arr.ndim == 1:
                # Prefer float32 for Gaussian-style attrs
                kind = "f4" if arr.dtype.kind == "f" else arr.dtype.str
                dtype_list.append((name, kind if kind.startswith(("f", "i", "u")) else "f4"))
                attr_arrays[name] = arr
            # skip multi-dim without flattening

    vertex = np.empty(n, dtype=dtype_list)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    if colors is not None and len(colors) == n:
        rgb = np.clip(colors, 0.0, 1.0)
        if rgb.max() <= 1.0:
            rgb_u8 = (rgb * 255.0).astype(np.uint8)
        else:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        vertex["red"] = rgb_u8[:, 0]
        vertex["green"] = rgb_u8[:, 1]
        vertex["blue"] = rgb_u8[:, 2]
    for name, arr in attr_arrays.items():
        vertex[name] = arr.astype(vertex.dtype.fields[name][0])

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def cloud_from_open3d(pcd: o3d.geometry.PointCloud, source: CloudData | None = None) -> CloudData:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    return CloudData(
        points=points,
        colors=colors if colors is not None and len(colors) else None,
        normals=normals if normals is not None and len(normals) else None,
        attributes={},
        is_gaussian=False,
        source_path=source.source_path if source else None,
    )
