"""Selectable object pipeline: segment → shape → semantics → per-object PLY export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .graph import GraphConfig
from .io import CloudData, load_cloud, write_ply
from .objects import ObjectConfig, ObjectResult, segment_objects
from .preprocess import PreprocessConfig, preprocess
from .segment import SegmentConfig, _palette_color
from .semantics import label_object, scene_extents
from .shape import ShapeHint, recognize_shape


@dataclass
class SelectablesConfig:
    input_path: str | Path
    output_dir: str | Path
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    uncenter_export: bool = True
    include_noise: bool = False
    write_labels_npy: bool = True


def _aabb(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
    }


def _subset_attrs(cloud: CloudData, indices: np.ndarray) -> dict[str, np.ndarray] | None:
    if not cloud.attributes:
        return None
    out = {k: v[indices] for k, v in cloud.attributes.items() if len(v) == len(cloud.points)}
    return out or None


def export_selectables(
    out_dir: Path,
    working: CloudData,
    center_offset: np.ndarray,
    result: ObjectResult,
    *,
    uncenter: bool = True,
    include_noise: bool = False,
    write_labels_npy: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    objects_dir = out_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    points = working.points.copy()
    if uncenter and center_offset is not None:
        points = points + center_offset

    scene = scene_extents(points)
    manifest_objects: list[dict[str, Any]] = []
    struct_i = 0
    obj_i = 0

    for cluster in result.objects:
        idx = cluster.indices
        pts = points[idx]
        shape = recognize_shape(pts)
        # For structure planes, prefer plane hint if RANSAC model exists
        if cluster.kind == "structure" and cluster.plane_model is not None:
            a, b, c, d = cluster.plane_model
            nrm = np.array([a, b, c], dtype=np.float64)
            nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
            shape = ShapeHint(
                "plane",
                max(shape.score, 0.8),
                {"normal": nrm.tolist(), "offset": float(d), "source": "ransac"},
            )

        semantic = label_object(pts, shape, scene, kind=cluster.kind)
        if cluster.kind == "structure":
            stem = f"structure_{struct_i:02d}"
            struct_i += 1
        else:
            stem = f"object_{obj_i:03d}"
            obj_i += 1

        rel = f"objects/{stem}.ply"
        col = np.tile(_palette_color(cluster.index), (len(idx), 1))
        write_ply(objects_dir / f"{stem}.ply", pts, col, _subset_attrs(working, idx))

        entry = {
            "id": stem,
            "kind": cluster.kind,
            "label": semantic,
            "shape": shape.shape,
            "shape_score": round(float(shape.score), 4),
            "shape_params": shape.params,
            "aabb": _aabb(pts),
            "centroid": pts.mean(axis=0).tolist(),
            "point_count": int(cluster.point_count),
            "path": rel,
        }
        if cluster.plane_model is not None:
            entry["plane_model"] = cluster.plane_model
        manifest_objects.append(entry)

    if include_noise and len(result.noise_indices):
        nidx = result.noise_indices
        write_ply(
            out_dir / "noise.ply",
            points[nidx],
            np.full((len(nidx), 3), 0.15),
            _subset_attrs(working, nidx),
        )

    if write_labels_npy:
        np.save(out_dir / "labels.npy", result.labels)

    # Colored overview
    colors = np.full((len(points), 3), 0.15)
    for cluster in result.objects:
        colors[cluster.indices] = _palette_color(cluster.index)
    attrs_all = working.attributes if working.attributes else None
    if attrs_all and any(len(v) != len(points) for v in attrs_all.values()):
        attrs_all = None
    write_ply(out_dir / "overview.ply", points, colors, attrs_all)

    manifest: dict[str, Any] = {
        "source": working.source_path,
        "is_gaussian": working.is_gaussian,
        "working_points": int(len(points)),
        "center_offset": center_offset.tolist() if center_offset is not None else [0, 0, 0],
        "scene": {
            "z_min": scene.z_min,
            "z_max": scene.z_max,
            "xy_span": scene.xy_span,
        },
        "object_count": len(manifest_objects),
        "structure_count": sum(1 for o in manifest_objects if o["kind"] == "structure"),
        "selectable_count": sum(1 for o in manifest_objects if o["kind"] == "object"),
        "noise_points": int(len(result.noise_indices)),
        "objects": manifest_objects,
        "lichtfeld_import": {
            "folder": "objects/",
            "instructions": (
                "Import each PLY in objects/ as a separate model/layer in LichtFeld Studio. "
                "Each file is independently selectable. Do not rely on labels.npy for groups."
            ),
        },
        "outputs": {
            "overview": "overview.ply",
            "manifest": "manifest.json",
            "labels": "labels.npy" if write_labels_npy else None,
            "noise": "noise.ply" if include_noise and len(result.noise_indices) else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_selectables(cfg: SelectablesConfig) -> dict[str, Any]:
    cloud = load_cloud(cfg.input_path)
    pre = preprocess(cloud, cfg.preprocess)
    if len(pre.pcd.points) == 0:
        raise RuntimeError("Preprocessing removed all points; relax outlier/voxel settings.")

    result = segment_objects(
        pre.pcd,
        cfg.objects,
        colors=pre.cloud.colors,
        attributes=pre.cloud.attributes,
    )
    summary = export_selectables(
        Path(cfg.output_dir),
        pre.cloud,
        pre.center_offset,
        result,
        uncenter=cfg.uncenter_export,
        include_noise=cfg.include_noise,
        write_labels_npy=cfg.write_labels_npy,
    )
    summary["config"] = {
        "input": str(Path(cfg.input_path).resolve()),
        "output": str(Path(cfg.output_dir).resolve()),
        "preprocess": cfg.preprocess.__dict__,
        "objects": {
            "method": cfg.objects.method,
            "min_object_points": cfg.objects.min_object_points,
            "max_objects": cfg.objects.max_objects,
            "graph_point_limit": cfg.objects.graph_point_limit,
            "residual_voxel": cfg.objects.residual_voxel,
            "segment": cfg.objects.segment.__dict__,
            "graph": cfg.objects.graph.__dict__,
        },
    }
    (Path(cfg.output_dir) / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def default_selectables_configs(
    *,
    skip_downsample: bool = True,
    voxel: float = 0.0,
    max_planes: int = 6,
    plane_dist: float = 0.03,
    min_object_points: int = 200,
    max_objects: int = 80,
    method: str = "graph",
) -> tuple[PreprocessConfig, ObjectConfig]:
    pre = PreprocessConfig(
        skip_downsample=skip_downsample or voxel <= 0,
        voxel_size=0.0 if skip_downsample else voxel,
    )
    obj = ObjectConfig(
        segment=SegmentConfig(
            max_planes=max_planes,
            distance_threshold=plane_dist,
            min_plane_inliers=80,
            # unused when method=graph but kept for dbscan fallback parity
            dbscan_eps=0.15,
            dbscan_min_points=20,
        ),
        graph=GraphConfig(),
        method=method,
        min_object_points=min_object_points,
        max_objects=max_objects,
    )
    return pre, obj
