"""Denoise, downsample, and normal estimation (Open3D; matches Spyder / RANSAC script)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .io import CloudData


@dataclass
class PreprocessConfig:
    center: bool = True
    outlier_nn: int = 16
    outlier_std: float = 2.0
    skip_outlier: bool = False
    voxel_size: float = 0.1
    skip_downsample: bool = False
    normal_knn: int = 16
    normal_radius_factor: float = 4.0
    skip_normals: bool = False


@dataclass
class PreprocessResult:
    cloud: CloudData
    pcd: o3d.geometry.PointCloud
    outlier_indices: np.ndarray
    center_offset: np.ndarray
    # Indices into the *original loaded* cloud for each working point (empty if lost).
    source_indices: np.ndarray


def preprocess(cloud: CloudData, cfg: PreprocessConfig) -> PreprocessResult:
    pcd = cloud.to_open3d()
    center_offset = np.zeros(3, dtype=np.float64)
    source_indices = np.arange(len(cloud.points), dtype=np.int64)

    if cfg.center:
        center_offset = np.asarray(pcd.get_center(), dtype=np.float64)
        pcd.translate(-center_offset)

    outlier_indices = np.array([], dtype=np.int64)
    if not cfg.skip_outlier and len(pcd.points) > cfg.outlier_nn:
        filtered, kept = pcd.remove_statistical_outlier(cfg.outlier_nn, cfg.outlier_std)
        kept = np.asarray(kept, dtype=np.int64)
        all_idx = np.arange(len(pcd.points))
        mask = np.ones(len(pcd.points), dtype=bool)
        mask[kept] = False
        outlier_indices = source_indices[mask]
        source_indices = source_indices[kept]
        pcd = filtered

    attributes_ok = True
    if not cfg.skip_downsample and cfg.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
        # Voxel downsample loses 1:1 provenance to source attributes.
        source_indices = np.array([], dtype=np.int64)
        attributes_ok = False

    if not cfg.skip_normals and len(pcd.points) > 0:
        nn_distance = float(np.mean(pcd.compute_nearest_neighbor_distance()))
        radius = max(nn_distance * cfg.normal_radius_factor, 1e-6)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=cfg.normal_knn
            )
        )

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    attrs: dict[str, np.ndarray] = {}
    if attributes_ok and cloud.attributes and len(source_indices) == len(points):
        attrs = {k: v[source_indices] for k, v in cloud.attributes.items()}
        # Prefer original RGB from Gaussian if present and downsample skipped
        if cloud.colors is not None and (colors is None or len(colors) == 0):
            colors = cloud.colors[source_indices]

    out = CloudData(
        points=points,
        colors=colors if colors is not None and len(colors) else None,
        normals=normals if normals is not None and len(normals) else None,
        attributes=attrs,
        is_gaussian=cloud.is_gaussian,
        source_path=cloud.source_path,
    )
    return PreprocessResult(
        cloud=out,
        pcd=pcd,
        outlier_indices=outlier_indices,
        center_offset=center_offset,
        source_indices=source_indices,
    )
