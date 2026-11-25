"""Smoke tests for seg3d pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from seg3d.io import load_cloud, write_ply
from seg3d.pipeline import PipelineConfig, run_pipeline
from seg3d.preprocess import PreprocessConfig
from seg3d.segment import SegmentConfig


def _synthetic_scene(path: Path) -> Path:
    rng = np.random.default_rng(0)
    # Floor plane z=0
    floor = np.column_stack(
        [rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000), rng.normal(0, 0.002, 2000)]
    )
    # Vertical wall x=1
    wall = np.column_stack(
        [
            rng.normal(1.0, 0.002, 1200),
            rng.uniform(-1, 1, 1200),
            rng.uniform(0, 1.2, 1200),
        ]
    )
    # Blob object
    blob = rng.normal([0.0, 0.0, 0.5], [0.08, 0.08, 0.1], size=(400, 3))
    pts = np.vstack([floor, wall, blob])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_run_pipeline_synthetic(tmp_path: Path) -> None:
    cloud_path = _synthetic_scene(tmp_path / "scene.ply")
    out = tmp_path / "out"
    summary = run_pipeline(
        PipelineConfig(
            input_path=cloud_path,
            output_dir=out,
            preprocess=PreprocessConfig(voxel_size=0.05, outlier_nn=16, outlier_std=2.0),
            segment=SegmentConfig(
                max_planes=3,
                distance_threshold=0.02,
                min_plane_inliers=80,
                dbscan_eps=0.1,
                dbscan_min_points=10,
            ),
        )
    )
    assert summary["working_points"] > 0
    assert len(summary["planes"]) >= 1
    assert (out / "segmented.ply").is_file()
    assert (out / "summary.json").is_file()
    loaded = load_cloud(out / "segmented.ply")
    assert len(loaded.points) == summary["working_points"]


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    pts = np.random.default_rng(1).random((100, 3))
    cols = np.zeros((100, 3))
    cols[:, 0] = 1.0
    path = tmp_path / "c.ply"
    write_ply(path, pts, cols, {"opacity": np.full(100, 0.5)})
    cloud = load_cloud(path)
    assert len(cloud.points) == 100
