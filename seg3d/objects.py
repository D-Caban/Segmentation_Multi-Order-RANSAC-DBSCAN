"""Object segmentation: structure planes + residual graph clustering with merge/cap."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .graph import GraphConfig, adaptive_eps, build_knn_graph, connected_components_by_affinity
from .segment import PlaneSegment, SegmentConfig, multi_order_ransac


@dataclass
class ObjectConfig:
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    # Prefer graph CC; fall back to adaptive DBSCAN when residual graph is huge.
    method: str = "graph"  # "graph" | "dbscan"
    graph_point_limit: int = 80_000
    residual_voxel: float = 0.0  # >0 forces voxel for residual clustering
    min_object_points: int = 200
    max_objects: int = 80  # residual objects (structure planes counted separately)
    dbscan_eps_factor: float = 2.5
    dbscan_min_points: int = 20
    merge_noise: bool = True


@dataclass
class ObjectCluster:
    index: int
    indices: np.ndarray  # into working cloud
    point_count: int
    kind: str  # "structure" | "object"
    plane_model: list[float] | None = None


@dataclass
class ObjectResult:
    objects: list[ObjectCluster] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    noise_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    planes: list[PlaneSegment] = field(default_factory=list)


def _opacity_from_attrs(attributes: dict[str, np.ndarray] | None, n: int) -> np.ndarray | None:
    if not attributes or "opacity" not in attributes:
        return None
    raw = np.asarray(attributes["opacity"], dtype=np.float64)
    if len(raw) != n:
        return None
    # Graphdeco often stores logit opacity; map roughly to (0,1) if needed.
    if np.nanmax(np.abs(raw)) > 1.5:
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
    return np.clip(raw, 0.0, 1.0)


def _voxel_cluster_then_propagate(
    residual_pts: np.ndarray,
    residual_map: np.ndarray,
    voxel: float,
    cluster_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster on voxel-downsampled residual; propagate labels via voxel keys + NN fallback.

    Returns (labels_on_full_residual, unique kept label ids >=0).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(residual_pts)
    down = pcd.voxel_down_sample(voxel)
    down_pts = np.asarray(down.points)
    if len(down_pts) < 2:
        return np.full(len(residual_pts), -1, dtype=np.int64), np.array([], dtype=np.int64)

    down_labels = cluster_fn(down_pts)
    # Fast path: same voxel key → same label as the downsampled representative.
    v = max(float(voxel), 1e-9)
    down_keys = np.floor(down_pts / v).astype(np.int64)
    key_to_label: dict[tuple[int, int, int], int] = {}
    for key, lab in zip(down_keys, down_labels):
        key_to_label[tuple(int(x) for x in key)] = int(lab)

    full_keys = np.floor(residual_pts / v).astype(np.int64)
    full_labels = np.full(len(residual_pts), -1, dtype=np.int64)
    missing = []
    for i, key in enumerate(full_keys):
        lab = key_to_label.get(tuple(int(x) for x in key))
        if lab is None:
            missing.append(i)
        else:
            full_labels[i] = lab

    if missing:
        tree = o3d.geometry.KDTreeFlann(down)
        for i in missing:
            _, idx, _ = tree.search_knn_vector_3d(residual_pts[i], 1)
            full_labels[i] = int(down_labels[idx[0]])

    kept = np.unique(full_labels[full_labels >= 0])
    return full_labels, kept


def _dbscan_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points), dtype=np.int64)


def _split_oversized_with_dbscan(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_fraction: float = 0.25,
    eps_factor: float = 2.0,
    min_points: int = 20,
    subsample_cap: int = 60_000,
) -> np.ndarray:
    """If one component owns too much of the residual, DBSCAN-split it (subsample if huge)."""
    out = labels.copy()
    n = len(points)
    if n == 0:
        return out
    ids = [int(i) for i in np.unique(out) if i >= 0]
    if not ids:
        return out
    next_id = max(ids) + 1
    rng = np.random.default_rng(0)

    for lid in list(ids):
        mask = out == lid
        count = int(mask.sum())
        if count < max(min_points * 4, 100) or count / n < max_fraction:
            continue
        idxs = np.flatnonzero(mask)
        sub = points[mask]
        eps = adaptive_eps(sub[:: max(1, len(sub) // 20000)], knn=8, factor=eps_factor)

        if len(sub) > subsample_cap:
            pick = rng.choice(len(sub), size=subsample_cap, replace=False)
            sample = sub[pick]
            sample_labels = _dbscan_labels(sample, eps, min_points)
            if len(np.unique(sample_labels[sample_labels >= 0])) < 2:
                # Tighten eps once and retry
                sample_labels = _dbscan_labels(sample, eps * 0.6, min_points)
            if len(np.unique(sample_labels[sample_labels >= 0])) < 2:
                continue
            # Propagate sample labels → all points in this component
            tree_pcd = o3d.geometry.PointCloud()
            tree_pcd.points = o3d.utility.Vector3dVector(sample)
            tree = o3d.geometry.KDTreeFlann(tree_pcd)
            sub_labels = np.full(len(sub), -1, dtype=np.int64)
            for i in range(len(sub)):
                _, nn, _ = tree.search_knn_vector_3d(sub[i], 1)
                sub_labels[i] = int(sample_labels[nn[0]])
        else:
            sub_labels = _dbscan_labels(sub, eps, min_points)
            if len(np.unique(sub_labels[sub_labels >= 0])) < 2:
                sub_labels = _dbscan_labels(sub, eps * 0.6, min_points)
            if len(np.unique(sub_labels[sub_labels >= 0])) < 2:
                continue

        pos = [int(x) for x in np.unique(sub_labels) if x >= 0]
        if len(pos) < 2:
            continue
        # Map local subcluster ids → global labels; assign DBSCAN noise to nearest subcluster.
        centers = {sid: sub[sub_labels == sid].mean(0) for sid in pos}
        center_ids = list(centers.keys())
        center_pts = np.stack([centers[s] for s in center_ids])
        first = True
        local_to_global: dict[int, int] = {}
        for sid in pos:
            if first:
                local_to_global[sid] = lid
                first = False
            else:
                local_to_global[sid] = next_id
                next_id += 1
        for i, sl in enumerate(sub_labels):
            if sl >= 0:
                out[idxs[i]] = local_to_global[int(sl)]
            else:
                d = np.linalg.norm(center_pts - sub[i], axis=1)
                out[idxs[i]] = local_to_global[center_ids[int(np.argmin(d))]]
    return out


def _cluster_residuals_graph(
    residual_pts: np.ndarray,
    residual_map: np.ndarray,
    cfg: ObjectConfig,
    *,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    opacity: np.ndarray | None,
) -> np.ndarray:
    """Return labels aligned with residual_pts (-1 = noise/singleton discard later)."""
    n = len(residual_pts)
    if n == 0:
        return np.array([], dtype=np.int64)

    voxel = cfg.residual_voxel
    if voxel <= 0 and n > cfg.graph_point_limit:
        # Auto voxel from median spacing estimate
        eps = adaptive_eps(residual_pts[:: max(1, n // 20000)], knn=6, factor=1.0)
        voxel = max(eps * 1.5, 1e-3)

    def graph_labels(pts: np.ndarray) -> np.ndarray:
        # Subsample cues to matching length when voxelized (cues None then)
        g = build_knn_graph(pts, cfg.graph, normals=None, colors=None, opacity=None)
        labels = connected_components_by_affinity(g, cfg=cfg.graph)
        return _split_oversized_with_dbscan(
            pts,
            labels,
            eps_factor=cfg.dbscan_eps_factor,
            min_points=cfg.dbscan_min_points,
        )

    if voxel > 0 and n > 500:
        # Split oversized components on the voxel graph, then NN-propagate.
        # Avoid a second DBSCAN on the full residual (marks most points as noise).
        labels, _ = _voxel_cluster_then_propagate(residual_pts, residual_map, voxel, graph_labels)
        return labels

    # Full-res graph (may be heavy); allow cue arrays
    g = build_knn_graph(residual_pts, cfg.graph, normals=normals, colors=colors, opacity=opacity)
    labels = connected_components_by_affinity(g, cfg=cfg.graph)
    return _split_oversized_with_dbscan(
        residual_pts,
        labels,
        max_fraction=0.2,
        eps_factor=cfg.dbscan_eps_factor,
        min_points=cfg.dbscan_min_points,
        subsample_cap=min(cfg.graph_point_limit, 60_000),
    )


def _cluster_residuals_dbscan(residual_pts: np.ndarray, cfg: ObjectConfig) -> np.ndarray:
    if len(residual_pts) == 0:
        return np.array([], dtype=np.int64)
    eps = adaptive_eps(residual_pts, knn=8, factor=cfg.dbscan_eps_factor)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(residual_pts)
    return np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=cfg.dbscan_min_points),
        dtype=np.int64,
    )


def _merge_small_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    min_points: int,
    *,
    max_merge_dist: float | None = None,
) -> np.ndarray:
    """Merge clusters smaller than min_points into nearest larger centroid; else → -1."""
    out = labels.copy()
    valid = np.unique(out[out >= 0])
    if len(valid) == 0:
        return out

    centroids: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for lid in valid:
        mask = out == lid
        counts[int(lid)] = int(mask.sum())
        centroids[int(lid)] = points[mask].mean(axis=0)

    large = [lid for lid, c in counts.items() if c >= min_points]
    small = [lid for lid, c in counts.items() if c < min_points]
    if not large:
        # Everything small — keep the biggest few as-is, rest noise
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        keep = {ordered[0][0]} if ordered else set()
        for lid in valid:
            if int(lid) not in keep:
                out[out == lid] = -1
        return out

    # Default merge radius: fraction of scene diagonal
    if max_merge_dist is None:
        extent = float(np.linalg.norm(points.max(0) - points.min(0)))
        max_merge_dist = max(0.15 * extent, 1e-3)

    large_pts = np.stack([centroids[i] for i in large])
    large_ids = np.asarray(large, dtype=np.int64)
    for lid in small:
        c = centroids[lid]
        d = np.linalg.norm(large_pts - c, axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= max_merge_dist:
            out[out == lid] = int(large_ids[j])
        else:
            out[out == lid] = -1
    return out


def _cap_objects(points: np.ndarray, labels: np.ndarray, max_objects: int) -> np.ndarray:
    out = labels.copy()
    while True:
        ids = [int(i) for i in np.unique(out) if i >= 0]
        if len(ids) <= max_objects:
            break
        counts = {i: int((out == i).sum()) for i in ids}
        smallest = min(ids, key=lambda i: counts[i])
        centroids = {i: points[out == i].mean(0) for i in ids if i != smallest}
        if not centroids:
            break
        c = points[out == smallest].mean(0)
        others = list(centroids.keys())
        d = [float(np.linalg.norm(centroids[j] - c)) for j in others]
        target = others[int(np.argmin(d))]
        out[out == smallest] = target
    return out


def _relabel_dense(labels: np.ndarray) -> np.ndarray:
    out = np.full_like(labels, -1)
    next_id = 0
    for lid in sorted(int(x) for x in np.unique(labels) if x >= 0):
        out[labels == lid] = next_id
        next_id += 1
    return out


def segment_objects(
    pcd: o3d.geometry.PointCloud,
    cfg: ObjectConfig | None = None,
    *,
    colors: np.ndarray | None = None,
    attributes: dict[str, np.ndarray] | None = None,
) -> ObjectResult:
    """Extract structure planes + residual object clusters."""
    cfg = cfg or ObjectConfig()
    n = len(pcd.points)
    working_labels = np.full(n, -1, dtype=np.int64)
    points = np.asarray(pcd.points)

    planes, residual, residual_map = multi_order_ransac(pcd, cfg.segment)
    objects: list[ObjectCluster] = []
    for plane in planes:
        working_labels[plane.indices] = plane.index
        objects.append(
            ObjectCluster(
                index=plane.index,
                indices=plane.indices,
                point_count=plane.point_count,
                kind="structure",
                plane_model=plane.model,
            )
        )

    if len(residual.points) == 0:
        return ObjectResult(objects=objects, labels=working_labels, noise_indices=np.array([], dtype=np.int64), planes=planes)

    residual_pts = np.asarray(residual.points)
    res_normals = np.asarray(residual.normals) if residual.has_normals() else None
    res_colors = None
    if colors is not None and len(colors) == n:
        res_colors = colors[residual_map]
    res_opacity = _opacity_from_attrs(attributes, n)
    if res_opacity is not None:
        res_opacity = res_opacity[residual_map]

    if cfg.method == "dbscan":
        local_labels = _cluster_residuals_dbscan(residual_pts, cfg)
    else:
        local_labels = _cluster_residuals_graph(
            residual_pts,
            residual_map,
            cfg,
            normals=res_normals,
            colors=res_colors,
            opacity=res_opacity,
        )

    local_labels = _merge_small_clusters(residual_pts, local_labels, cfg.min_object_points)
    local_labels = _cap_objects(residual_pts, local_labels, cfg.max_objects)
    local_labels = _relabel_dense(local_labels)

    first = len(planes)
    noise_local = local_labels < 0
    noise_idx = residual_map[noise_local]

    for lid in sorted(int(x) for x in np.unique(local_labels) if x >= 0):
        members = np.flatnonzero(local_labels == lid)
        global_idx = residual_map[members]
        oid = first + lid
        working_labels[global_idx] = oid
        objects.append(
            ObjectCluster(
                index=oid,
                indices=global_idx,
                point_count=len(global_idx),
                kind="object",
            )
        )

    return ObjectResult(
        objects=objects,
        labels=working_labels,
        noise_indices=noise_idx,
        planes=planes,
    )
