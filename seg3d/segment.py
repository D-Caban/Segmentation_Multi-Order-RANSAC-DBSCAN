"""Multi-order RANSAC plane extraction and residual DBSCAN clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


@dataclass
class SegmentConfig:
    max_planes: int = 6
    distance_threshold: float = 0.02
    ransac_n: int = 3
    num_iterations: int = 1000
    min_plane_inliers: int = 50
    dbscan_eps: float = 0.05
    dbscan_min_points: int = 5
    skip_dbscan: bool = False


@dataclass
class PlaneSegment:
    index: int
    model: list[float]  # [a, b, c, d]
    indices: np.ndarray  # indices into working cloud
    point_count: int


@dataclass
class ClusterSegment:
    index: int
    indices: np.ndarray
    point_count: int


@dataclass
class SegmentResult:
    planes: list[PlaneSegment] = field(default_factory=list)
    clusters: list[ClusterSegment] = field(default_factory=list)
    residual_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # labels: >=0 plane id, or plane_count + cluster_id, or -1 noise


def _palette_color(i: int) -> list[float]:
    # Stable distinct-ish colors without requiring matplotlib at import time
    golden = 0.61803398875
    h = (i * golden) % 1.0
    # simple HSV→RGB
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return [r, g, b]


def multi_order_ransac(
    pcd: o3d.geometry.PointCloud, cfg: SegmentConfig
) -> tuple[list[PlaneSegment], o3d.geometry.PointCloud, np.ndarray]:
    """Iteratively extract planes. Returns planes, residual cloud, residual index map."""
    remaining = pcd
    # Map from current remaining indices → original working-cloud indices
    index_map = np.arange(len(pcd.points), dtype=np.int64)
    planes: list[PlaneSegment] = []

    for i in range(cfg.max_planes):
        if len(remaining.points) < max(cfg.min_plane_inliers, cfg.ransac_n):
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=cfg.distance_threshold,
            ransac_n=cfg.ransac_n,
            num_iterations=cfg.num_iterations,
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) < cfg.min_plane_inliers:
            break

        original_idx = index_map[inliers]
        planes.append(
            PlaneSegment(
                index=i,
                model=[float(x) for x in model],
                indices=original_idx,
                point_count=len(original_idx),
            )
        )
        remaining = remaining.select_by_index(inliers.tolist(), invert=True)
        mask = np.ones(len(index_map), dtype=bool)
        mask[inliers] = False
        index_map = index_map[mask]

    return planes, remaining, index_map


def cluster_residuals(
    residual: o3d.geometry.PointCloud,
    residual_index_map: np.ndarray,
    cfg: SegmentConfig,
    first_label: int,
) -> tuple[list[ClusterSegment], np.ndarray, np.ndarray]:
    """DBSCAN residual points. Returns clusters, noise original indices, label array for residual."""
    if cfg.skip_dbscan or len(residual.points) == 0:
        return [], residual_index_map.copy(), np.full(len(residual_index_map), -1, dtype=np.int64)

    labels = np.asarray(
        residual.cluster_dbscan(eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points),
        dtype=np.int64,
    )
    clusters: list[ClusterSegment] = []
    for local_id in sorted(set(labels.tolist())):
        if local_id < 0:
            continue
        members = np.flatnonzero(labels == local_id)
        clusters.append(
            ClusterSegment(
                index=first_label + len(clusters),
                indices=residual_index_map[members],
                point_count=len(members),
            )
        )
    # Remap local DBSCAN ids → global cluster label indices (first_label+)
    remapped = np.full(len(labels), -1, dtype=np.int64)
    next_id = first_label
    for local_id in sorted(set(labels.tolist())):
        if local_id < 0:
            continue
        remapped[labels == local_id] = next_id
        next_id += 1
    noise = residual_index_map[labels < 0]
    return clusters, noise, remapped


def segment(pcd: o3d.geometry.PointCloud, cfg: SegmentConfig) -> SegmentResult:
    n = len(pcd.points)
    labels = np.full(n, -1, dtype=np.int64)

    planes, residual, residual_map = multi_order_ransac(pcd, cfg)
    for plane in planes:
        labels[plane.indices] = plane.index

    clusters, noise_idx, _ = cluster_residuals(
        residual, residual_map, cfg, first_label=len(planes)
    )
    for cluster in clusters:
        labels[cluster.indices] = cluster.index

    return SegmentResult(
        planes=planes,
        clusters=clusters,
        residual_indices=noise_idx,
        labels=labels,
    )


def colorize_by_labels(pcd: o3d.geometry.PointCloud, labels: np.ndarray) -> o3d.geometry.PointCloud:
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    for lid in np.unique(labels):
        if lid < 0:
            colors[labels < 0] = [0.15, 0.15, 0.15]
        else:
            colors[labels == lid] = _palette_color(int(lid))
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out
