"""Export labeled clouds, planes, and clusters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import CloudData, write_ply
from .preprocess import PreprocessResult
from .segment import SegmentResult, _palette_color


def _subset_attrs(cloud: CloudData, indices: np.ndarray) -> dict[str, np.ndarray]:
    if not cloud.attributes:
        return {}
    return {k: v[indices] for k, v in cloud.attributes.items() if len(v) == len(cloud.points)}


def export_results(
    out_dir: Path,
    working: CloudData,
    preprocess: PreprocessResult,
    segments: SegmentResult,
    *,
    uncenter: bool = True,
) -> dict[str, Any]:
    """Write segmented PLYs and a JSON summary. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = working.points.copy()
    if uncenter and preprocess.center_offset is not None:
        points = points + preprocess.center_offset

    labels = segments.labels
    colors = np.zeros((len(points), 3), dtype=np.float64)
    for lid in np.unique(labels):
        if lid < 0:
            colors[labels < 0] = [0.15, 0.15, 0.15]
        else:
            colors[labels == lid] = _palette_color(int(lid))

    # Full labeled cloud (geometry + label colors; Gaussian attrs if 1:1 with working)
    attrs_all = working.attributes if working.attributes else None
    # Only attach attrs when they match working point count (true after no downsample
    # mismatch — after voxel downsample attrs are dropped in preprocess).
    if attrs_all and any(len(v) != len(points) for v in attrs_all.values()):
        attrs_all = None

    write_ply(out_dir / "segmented.ply", points, colors, attrs_all)

    plane_dir = out_dir / "planes"
    cluster_dir = out_dir / "clusters"
    plane_dir.mkdir(exist_ok=True)
    cluster_dir.mkdir(exist_ok=True)

    plane_summaries = []
    for plane in segments.planes:
        idx = plane.indices
        col = np.tile(_palette_color(plane.index), (len(idx), 1))
        write_ply(
            plane_dir / f"plane_{plane.index:02d}.ply",
            points[idx],
            col,
            _subset_attrs(working, idx) or None,
        )
        a, b, c, d = plane.model
        plane_summaries.append(
            {
                "id": plane.index,
                "equation": {"a": a, "b": b, "c": c, "d": d},
                "point_count": plane.point_count,
                "path": f"planes/plane_{plane.index:02d}.ply",
            }
        )

    cluster_summaries = []
    for cluster in segments.clusters:
        idx = cluster.indices
        col = np.tile(_palette_color(cluster.index), (len(idx), 1))
        write_ply(
            cluster_dir / f"cluster_{cluster.index:02d}.ply",
            points[idx],
            col,
            _subset_attrs(working, idx) or None,
        )
        cluster_summaries.append(
            {
                "id": cluster.index,
                "point_count": cluster.point_count,
                "path": f"clusters/cluster_{cluster.index:02d}.ply",
            }
        )

    noise_idx = segments.residual_indices
    if len(noise_idx):
        write_ply(
            out_dir / "noise.ply",
            points[noise_idx],
            np.full((len(noise_idx), 3), 0.15),
            _subset_attrs(working, noise_idx) or None,
        )

    np.save(out_dir / "labels.npy", labels)

    summary: dict[str, Any] = {
        "source": working.source_path,
        "is_gaussian": working.is_gaussian,
        "working_points": int(len(points)),
        "center_offset": preprocess.center_offset.tolist(),
        "planes": plane_summaries,
        "clusters": cluster_summaries,
        "noise_points": int(len(noise_idx)),
        "outputs": {
            "segmented": "segmented.ply",
            "labels": "labels.npy",
            "noise": "noise.ply" if len(noise_idx) else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
