"""Neighborhood graph on Gaussian / point centers (SplatGraph-style affinity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class GraphConfig:
    knn: int = 12
    use_normals: bool = True
    use_colors: bool = True
    use_opacity: bool = True
    # Soft weights in [0, 1] multiplied into affinity; 0 disables that cue.
    normal_weight: float = 0.35
    color_weight: float = 0.25
    opacity_weight: float = 0.15
    # Keep edges with affinity >= this quantile of observed affinities,
    # or absolute min_affinity if set > 0.
    affinity_quantile: float = 0.55
    min_affinity: float = 0.0
    # Drop edges longer than max_dist_factor × median kNN distance (0 = disable).
    max_dist_factor: float = 2.0


@dataclass
class NeighborhoodGraph:
    """Undirected kNN graph: for each node i, neighbors[i] and affinities[i] align."""

    neighbors: list[np.ndarray]
    affinities: list[np.ndarray]
    distances: list[np.ndarray]
    median_nn_dist: float
    n: int


def _pairwise_affinity(
    i: int,
    j: int,
    dist: float,
    scale: float,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    opacity: np.ndarray | None,
    cfg: GraphConfig,
) -> float:
    # Distance kernel: 1 at 0, ~0 beyond a few scales.
    s = max(scale, 1e-9)
    w = float(np.exp(-dist / s))
    if cfg.use_normals and normals is not None and cfg.normal_weight > 0:
        ni = normals[i]
        nj = normals[j]
        nn = float(np.linalg.norm(ni) * np.linalg.norm(nj))
        if nn > 1e-12:
            cos = abs(float(np.dot(ni, nj) / nn))
            w *= (1.0 - cfg.normal_weight) + cfg.normal_weight * cos
    if cfg.use_colors and colors is not None and cfg.color_weight > 0:
        dcol = float(np.linalg.norm(colors[i] - colors[j]))
        # RGB in [0,1]; max chord ~sqrt(3)
        csim = max(0.0, 1.0 - dcol / 1.7320508)
        w *= (1.0 - cfg.color_weight) + cfg.color_weight * csim
    if cfg.use_opacity and opacity is not None and cfg.opacity_weight > 0:
        osim = 1.0 - abs(float(opacity[i] - opacity[j]))
        w *= (1.0 - cfg.opacity_weight) + cfg.opacity_weight * max(0.0, osim)
    return w


def build_knn_graph(
    points: np.ndarray,
    cfg: GraphConfig | None = None,
    *,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
) -> NeighborhoodGraph:
    """Build a symmetric-ish kNN affinity graph with Open3D KDTreeFlann."""
    cfg = cfg or GraphConfig()
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return NeighborhoodGraph([], [], [], 1.0, 0)

    k = int(max(1, min(cfg.knn, n - 1))) if n > 1 else 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    tree = o3d.geometry.KDTreeFlann(pcd)

    neighbors: list[list[int]] = [[] for _ in range(n)]
    raw_dists: list[list[float]] = [[] for _ in range(n)]
    nn_dists: list[float] = []

    for i in range(n):
        if k <= 0:
            break
        _, idx, dist2 = tree.search_knn_vector_3d(pcd.points[i], k + 1)
        for j, d2 in zip(idx[1:], dist2[1:]):
            j = int(j)
            if j == i:
                continue
            dist = float(np.sqrt(max(d2, 0.0)))
            nn_dists.append(dist)
            neighbors[i].append(j)
            raw_dists[i].append(dist)

    median_nn = float(np.median(nn_dists)) if nn_dists else 1.0
    if not np.isfinite(median_nn) or median_nn <= 0:
        median_nn = 1.0
    max_dist = median_nn * cfg.max_dist_factor if cfg.max_dist_factor > 0 else float("inf")

    # Affinities with scale + optional cues; undirected union; distance gate.
    undirected: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n):
        for j, dist in zip(neighbors[i], raw_dists[i]):
            if dist > max_dist:
                continue
            a, b = (i, j) if i < j else (j, i)
            aff = _pairwise_affinity(i, j, dist, median_nn, normals, colors, opacity, cfg)
            prev = undirected.get((a, b))
            if prev is None or aff > prev[0]:
                undirected[(a, b)] = (aff, dist)

    buckets_n: list[list[int]] = [[] for _ in range(n)]
    buckets_a: list[list[float]] = [[] for _ in range(n)]
    buckets_d: list[list[float]] = [[] for _ in range(n)]
    for (a, b), (aff, dist) in undirected.items():
        buckets_n[a].append(b)
        buckets_a[a].append(aff)
        buckets_d[a].append(dist)
        buckets_n[b].append(a)
        buckets_a[b].append(aff)
        buckets_d[b].append(dist)
    out_n = [np.asarray(buckets_n[i], dtype=np.int64) for i in range(n)]
    out_a = [np.asarray(buckets_a[i], dtype=np.float64) for i in range(n)]
    out_d = [np.asarray(buckets_d[i], dtype=np.float64) for i in range(n)]

    return NeighborhoodGraph(out_n, out_a, out_d, median_nn, n)


def connected_components_by_affinity(
    graph: NeighborhoodGraph,
    *,
    affinity_threshold: float | None = None,
    cfg: GraphConfig | None = None,
) -> np.ndarray:
    """Union-Find labels (-1 unused). Nodes with no kept edges become singletons then noise-merge later."""
    cfg = cfg or GraphConfig()
    n = graph.n
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    all_aff = np.concatenate([a for a in graph.affinities if len(a)]) if n else np.array([])
    if affinity_threshold is None:
        if cfg.min_affinity > 0:
            affinity_threshold = cfg.min_affinity
        elif len(all_aff):
            affinity_threshold = float(np.quantile(all_aff, cfg.affinity_quantile))
        else:
            affinity_threshold = 0.0

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j, aff in zip(graph.neighbors[i], graph.affinities[i]):
            if i < j and aff >= affinity_threshold:
                union(i, j)

    root_to_label: dict[int, int] = {}
    next_id = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_id
            next_id += 1
        labels[i] = root_to_label[r]
    return labels


def adaptive_eps(points: np.ndarray, knn: int = 8, factor: float = 2.5) -> float:
    """Median kNN distance × factor — used when falling back to Open3D DBSCAN."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return 0.05
    k = int(max(1, min(knn, n - 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists: list[float] = []
    step = max(1, n // 5000)  # subsample for speed on large clouds
    for i in range(0, n, step):
        _, _, dist2 = tree.search_knn_vector_3d(pcd.points[i], k + 1)
        if len(dist2) > 1:
            dists.append(float(np.sqrt(max(dist2[1], 0.0))))
    med = float(np.median(dists)) if dists else 0.05
    return max(med * factor, 1e-4)
