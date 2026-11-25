"""CLI entrypoint: seg3d run | selectables | objects …"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import GraphConfig
from .objects import ObjectConfig
from .pipeline import PipelineConfig, run_pipeline
from .preprocess import PreprocessConfig
from .segment import SegmentConfig
from .selectables import SelectablesConfig, run_selectables


def _add_preprocess_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-center", action="store_true", help="Do not translate cloud to origin")
    p.add_argument("--skip-outlier", action="store_true")
    p.add_argument("--outlier-nn", type=int, default=16)
    p.add_argument("--outlier-std", type=float, default=2.0)
    p.add_argument("--voxel", type=float, default=0.1, help="Voxel size for downsample (0 to skip)")
    p.add_argument("--skip-downsample", action="store_true", help="Keep all points (preserves Gaussian attrs)")
    p.add_argument("--skip-normals", action="store_true")


def _add_plane_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-planes", type=int, default=6)
    p.add_argument("--plane-dist", type=float, default=0.02, help="RANSAC point-to-plane distance")
    p.add_argument("--ransac-n", type=int, default=3)
    p.add_argument("--ransac-iters", type=int, default=1000)
    p.add_argument("--min-inliers", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seg3d",
        description="Segment point clouds / Gaussian splat PLYs (planes, objects, selectables)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run multi-order RANSAC + residual DBSCAN")
    run.add_argument("--input", "-i", required=True, help="Input .ply / .pcd (or Gaussian training PLY)")
    run.add_argument("--out", "-o", required=True, help="Output directory")
    _add_preprocess_args(run)
    _add_plane_args(run)
    run.add_argument("--dbscan-eps", type=float, default=0.05)
    run.add_argument("--dbscan-min-points", type=int, default=5)
    run.add_argument("--skip-dbscan", action="store_true")
    run.add_argument("--keep-centered", action="store_true", help="Export in centered coordinates")

    for name, help_text in (
        ("selectables", "Export selectable per-object Gaussian PLYs + manifest"),
        ("objects", "Alias for selectables"),
    ):
        sel = sub.add_parser(name, help=help_text)
        sel.add_argument("--input", "-i", required=True)
        sel.add_argument("--out", "-o", required=True)
        _add_preprocess_args(sel)
        _add_plane_args(sel)
        sel.add_argument(
            "--method",
            choices=("graph", "dbscan"),
            default="graph",
            help="Residual clustering: affinity graph CC (default) or adaptive DBSCAN",
        )
        sel.add_argument("--min-object-points", type=int, default=200)
        sel.add_argument("--max-objects", type=int, default=80, help="Cap residual object count")
        sel.add_argument("--graph-knn", type=int, default=12)
        sel.add_argument("--graph-point-limit", type=int, default=80_000)
        sel.add_argument(
            "--residual-voxel",
            type=float,
            default=0.0,
            help="Voxel size for residual clustering (0 = auto when over point limit)",
        )
        sel.add_argument("--dbscan-eps-factor", type=float, default=2.5)
        sel.add_argument("--dbscan-min-points", type=int, default=20)
        sel.add_argument("--include-noise", action="store_true")
        sel.add_argument("--keep-centered", action="store_true")
        sel.add_argument("--no-labels-npy", action="store_true")

    return p


def _preprocess_from_args(args: argparse.Namespace) -> PreprocessConfig:
    voxel = 0.0 if args.skip_downsample else args.voxel
    return PreprocessConfig(
        center=not args.no_center,
        outlier_nn=args.outlier_nn,
        outlier_std=args.outlier_std,
        skip_outlier=args.skip_outlier,
        voxel_size=voxel,
        skip_downsample=args.skip_downsample or args.voxel <= 0,
        skip_normals=args.skip_normals,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        pre = _preprocess_from_args(args)
        seg = SegmentConfig(
            max_planes=args.max_planes,
            distance_threshold=args.plane_dist,
            ransac_n=args.ransac_n,
            num_iterations=args.ransac_iters,
            min_plane_inliers=args.min_inliers,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_points=args.dbscan_min_points,
            skip_dbscan=args.skip_dbscan,
        )
        cfg = PipelineConfig(
            input_path=args.input,
            output_dir=args.out,
            preprocess=pre,
            segment=seg,
            uncenter_export=not args.keep_centered,
        )
        summary = run_pipeline(cfg)
        brief = {
            "working_points": summary["working_points"],
            "planes": len(summary["planes"]),
            "clusters": len(summary["clusters"]),
            "noise_points": summary["noise_points"],
            "out": str(args.out),
            "is_gaussian": summary["is_gaussian"],
        }
        print(json.dumps(brief, indent=2))
        return 0

    if args.command in {"selectables", "objects"}:
        pre = _preprocess_from_args(args)
        # Selectables defaults: slightly looser plane_dist for indoor splats if user left 0.02
        plane_dist = args.plane_dist
        obj = ObjectConfig(
            segment=SegmentConfig(
                max_planes=args.max_planes,
                distance_threshold=plane_dist,
                ransac_n=args.ransac_n,
                num_iterations=args.ransac_iters,
                min_plane_inliers=args.min_inliers,
                dbscan_eps=0.15,
                dbscan_min_points=args.dbscan_min_points,
            ),
            graph=GraphConfig(knn=args.graph_knn),
            method=args.method,
            graph_point_limit=args.graph_point_limit,
            residual_voxel=args.residual_voxel,
            min_object_points=args.min_object_points,
            max_objects=args.max_objects,
            dbscan_eps_factor=args.dbscan_eps_factor,
            dbscan_min_points=args.dbscan_min_points,
        )
        cfg = SelectablesConfig(
            input_path=args.input,
            output_dir=args.out,
            preprocess=pre,
            objects=obj,
            uncenter_export=not args.keep_centered,
            include_noise=args.include_noise,
            write_labels_npy=not args.no_labels_npy,
        )
        summary = run_selectables(cfg)
        brief = {
            "working_points": summary["working_points"],
            "object_count": summary["object_count"],
            "structure_count": summary["structure_count"],
            "selectable_count": summary["selectable_count"],
            "noise_points": summary["noise_points"],
            "out": str(args.out),
            "is_gaussian": summary["is_gaussian"],
            "manifest": str(Path(args.out) / "manifest.json"),
        }
        print(json.dumps(brief, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
