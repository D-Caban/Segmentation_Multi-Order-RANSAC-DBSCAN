"""Coarse 3D shape recognition hints per object cluster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import open3d as o3d

ShapeName = Literal["plane", "box", "sphere", "cylinder", "unknown"]


@dataclass
class ShapeHint:
    shape: ShapeName
    score: float
    params: dict


def _plane_score(points: np.ndarray) -> tuple[float, dict]:
    if len(points) < 3:
        return 0.0, {}
    centered = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    d = -float(normal @ points.mean(axis=0))
    dist = np.abs(points @ normal + d)
    extent = float(np.linalg.norm(points.max(0) - points.min(0))) + 1e-9
    rms = float(np.sqrt(np.mean(dist**2)))
    # Flatness: smallest singular relative to largest
    flat = float(1.0 - singular[-1] / max(singular[0], 1e-12)) if len(singular) >= 3 else 0.0
    score = float(np.clip(0.55 * flat + 0.45 * (1.0 - min(rms / (0.05 * extent), 1.0)), 0.0, 1.0))
    return score, {"normal": normal.tolist(), "offset": d, "rms": rms}


def _box_score(points: np.ndarray) -> tuple[float, dict]:
    if len(points) < 4:
        return 0.0, {}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        obb = pcd.get_minimal_oriented_bounding_box()
    except Exception:
        aabb = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(aabb.get_extent(), dtype=np.float64)
        center = np.asarray(aabb.get_center(), dtype=np.float64)
        return 0.35, {"kind": "aabb", "center": center.tolist(), "extent": extent.tolist()}

    center = np.asarray(obb.center, dtype=np.float64)
    extent = np.asarray(obb.extent, dtype=np.float64)
    R = np.asarray(obb.R, dtype=np.float64)
    local = (points - center) @ R
    half = extent * 0.5
    # Distance outside OBB faces
    outside = np.maximum(np.abs(local) - half, 0.0)
    dist = np.linalg.norm(outside, axis=1)
    diag = float(np.linalg.norm(extent)) + 1e-9
    rms = float(np.sqrt(np.mean(dist**2)))
    fill = float(np.clip(1.0 - rms / (0.08 * diag), 0.0, 1.0))
    # Prefer non-degenerate boxes
    aspect = float(np.min(extent) / max(np.max(extent), 1e-12))
    score = float(np.clip(0.7 * fill + 0.3 * aspect, 0.0, 1.0))
    return score, {
        "kind": "obb",
        "center": center.tolist(),
        "extent": extent.tolist(),
        "rotation": R.tolist(),
        "rms": rms,
    }


def _sphere_score(points: np.ndarray) -> tuple[float, dict]:
    if len(points) < 6:
        return 0.0, {}
    # Algebraic fit: ||x - c||^2 = r^2 → 2 c·x + (r^2 - ||c||^2) = ||x||^2
    A = np.column_stack([2 * points, np.ones(len(points))])
    b = np.sum(points**2, axis=1)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, {}
    center = sol[:3]
    radius_sq = float(sol[3] + np.dot(center, center))
    if radius_sq <= 0:
        return 0.0, {}
    radius = float(np.sqrt(radius_sq))
    dist = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(dist**2)))
    score = float(np.clip(1.0 - rms / max(radius * 0.25, 1e-6), 0.0, 1.0))
    return score, {"center": center.tolist(), "radius": radius, "rms": rms}


def _cylinder_score(points: np.ndarray) -> tuple[float, dict]:
    if len(points) < 8:
        return 0.0, {}
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    # Radial residuals to mean radius in plane ⊥ axis
    proj = points - np.outer((points - points.mean(0)) @ axis, axis)
    # Actually: vector from mean to point, remove axis component
    v = points - points.mean(0)
    radial = v - np.outer(v @ axis, axis)
    radii = np.linalg.norm(radial, axis=1)
    r = float(np.median(radii))
    if r < 1e-6:
        return 0.0, {}
    dist = np.abs(radii - r)
    height = float(np.ptp(v @ axis))
    rms = float(np.sqrt(np.mean(dist**2)))
    score = float(np.clip(1.0 - rms / max(r * 0.3, 1e-6), 0.0, 1.0))
    # Prefer elongated cylinders
    if height < 1.5 * r:
        score *= 0.75
    return score, {
        "axis": axis.tolist(),
        "radius": r,
        "height": height,
        "center": points.mean(0).tolist(),
        "rms": rms,
    }


def recognize_shape(points: np.ndarray) -> ShapeHint:
    """Pick the best coarse primitive hint for a cluster."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return ShapeHint("unknown", 0.0, {})

    candidates: list[tuple[ShapeName, float, dict]] = [
        ("plane", *_plane_score(pts)),
        ("box", *_box_score(pts)),
        ("sphere", *_sphere_score(pts)),
        ("cylinder", *_cylinder_score(pts)),
    ]
    best_name, best_score, best_params = max(candidates, key=lambda t: t[1])
    if best_score < 0.25:
        return ShapeHint("unknown", float(best_score), best_params)
    return ShapeHint(best_name, float(best_score), best_params)
