"""Rule-based coarse semantic labels from shape + size + height."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .shape import ShapeHint

SemanticLabel = Literal["floor", "wall", "ceiling", "furniture", "unknown"]


@dataclass
class SceneExtents:
    z_min: float
    z_max: float
    xy_span: float

    @property
    def height(self) -> float:
        return max(self.z_max - self.z_min, 1e-9)


def scene_extents(points: np.ndarray) -> SceneExtents:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return SceneExtents(0.0, 1.0, 1.0)
    z_min = float(pts[:, 2].min())
    z_max = float(pts[:, 2].max())
    xy = pts[:, :2]
    xy_span = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    return SceneExtents(z_min, z_max, max(xy_span, 1e-9))


def label_object(
    points: np.ndarray,
    shape: ShapeHint,
    scene: SceneExtents,
    *,
    kind: str = "object",
) -> SemanticLabel:
    """Assign a coarse semantic tag. Structure planes get architectural bias."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return "unknown"

    aabb_min = pts.min(0)
    aabb_max = pts.max(0)
    extent = aabb_max - aabb_min
    centroid = pts.mean(0)
    z_rel = (float(centroid[2]) - scene.z_min) / scene.height
    horiz_span = float(np.linalg.norm(extent[:2]))
    vert_span = float(extent[2])

    normal = None
    if shape.shape == "plane" and "normal" in shape.params:
        normal = np.asarray(shape.params["normal"], dtype=np.float64)
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        upright = abs(float(normal[2]))  # ~1 horizontal plane, ~0 vertical

        if upright > 0.75:
            # Horizontal plane
            if z_rel < 0.22 or float(aabb_min[2] - scene.z_min) / scene.height < 0.15:
                return "floor"
            if z_rel > 0.75 or float(aabb_max[2] - scene.z_min) / scene.height > 0.85:
                return "ceiling"
            # Mid-height slab — still call floor-like if very flat and wide
            if horiz_span > 0.4 * scene.xy_span and vert_span < 0.08 * scene.height:
                return "floor"
            return "furniture"
        if upright < 0.35:
            return "wall"

    if kind == "structure" and shape.shape == "plane":
        return "unknown"

    # Box / cylinder / sphere at mid height → furniture
    if shape.shape in {"box", "cylinder", "sphere"} and 0.05 < z_rel < 0.85:
        if vert_span > 0.03 * scene.height or horiz_span > 0.02 * scene.xy_span:
            return "furniture"

    # Large mid residual blob
    if 0.08 < z_rel < 0.8 and horiz_span > 0.03 * scene.xy_span:
        return "furniture"

    return "unknown"
