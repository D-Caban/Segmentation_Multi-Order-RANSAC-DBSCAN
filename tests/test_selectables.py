"""Tests for selectable object segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from seg3d.graph import GraphConfig, build_knn_graph, connected_components_by_affinity
from seg3d.io import load_cloud
from seg3d.objects import ObjectConfig, segment_objects
from seg3d.preprocess import PreprocessConfig
from seg3d.segment import SegmentConfig
from seg3d.selectables import SelectablesConfig, run_selectables
from seg3d.semantics import label_object, scene_extents
from seg3d.shape import recognize_shape


def _synthetic_house(path: Path) -> Path:
    rng = np.random.default_rng(0)
    floor = np.column_stack(
        [rng.uniform(-2, 2, 2500), rng.uniform(-2, 2, 2500), rng.normal(0, 0.002, 2500)]
    )
    wall = np.column_stack(
        [
            rng.normal(2.0, 0.003, 1500),
            rng.uniform(-2, 2, 1500),
            rng.uniform(0, 2.0, 1500),
        ]
    )
    # Two furniture-like blobs
    chair = rng.normal([0.0, 0.5, 0.45], [0.12, 0.12, 0.2], size=(500, 3))
    table = rng.normal([-0.8, -0.6, 0.4], [0.25, 0.18, 0.05], size=(600, 3))
    pts = np.vstack([floor, wall, chair, table])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_graph_connected_components() -> None:
    # Two separated blobs
    a = np.random.default_rng(1).normal([0, 0, 0], 0.05, size=(40, 3))
    b = np.random.default_rng(2).normal([3, 0, 0], 0.05, size=(40, 3))
    pts = np.vstack([a, b])
    g = build_knn_graph(pts, GraphConfig(knn=6, use_normals=False, use_colors=False, use_opacity=False))
    labels = connected_components_by_affinity(g, affinity_threshold=0.01)
    assert len(np.unique(labels)) >= 2


def test_shape_and_semantics_floor() -> None:
    rng = np.random.default_rng(3)
    floor = np.column_stack(
        [rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400), rng.normal(0, 0.002, 400)]
    )
    hint = recognize_shape(floor)
    assert hint.shape in {"plane", "box"}
    scene = scene_extents(floor)
    label = label_object(floor, hint, scene, kind="structure")
    assert label in {"floor", "unknown", "furniture"}


def test_run_selectables_synthetic(tmp_path: Path) -> None:
    cloud_path = _synthetic_house(tmp_path / "house.ply")
    out = tmp_path / "sel"
    summary = run_selectables(
        SelectablesConfig(
            input_path=cloud_path,
            output_dir=out,
            preprocess=PreprocessConfig(voxel_size=0.05, skip_downsample=False),
            objects=ObjectConfig(
                segment=SegmentConfig(
                    max_planes=3,
                    distance_threshold=0.025,
                    min_plane_inliers=100,
                ),
                method="dbscan",
                min_object_points=50,
                max_objects=20,
                dbscan_eps_factor=3.0,
                dbscan_min_points=10,
            ),
        )
    )
    assert summary["working_points"] > 0
    assert summary["object_count"] >= 1
    assert (out / "manifest.json").is_file()
    assert (out / "overview.ply").is_file()
    assert (out / "objects").is_dir()
    objs = list((out / "objects").glob("*.ply"))
    assert len(objs) == summary["object_count"]
    # Manifest paths exist
    for entry in summary["objects"]:
        assert (out / entry["path"]).is_file()
        assert "label" in entry
        assert "shape" in entry
        assert "aabb" in entry


def test_segment_objects_caps(tmp_path: Path) -> None:
    cloud_path = _synthetic_house(tmp_path / "h.ply")
    cloud = load_cloud(cloud_path)
    pcd = cloud.to_open3d()
    pcd.estimate_normals()
    result = segment_objects(
        pcd,
        ObjectConfig(
            segment=SegmentConfig(max_planes=2, distance_threshold=0.03, min_plane_inliers=80),
            method="dbscan",
            min_object_points=40,
            max_objects=5,
            dbscan_min_points=8,
        ),
    )
    residual_objs = [o for o in result.objects if o.kind == "object"]
    assert len(residual_objs) <= 5
