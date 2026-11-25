"""CLI entrypoint: seg3d run …"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import PipelineConfig, run_pipeline
from .preprocess import PreprocessConfig
from .segment import SegmentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seg3d",
        description="Segment point clouds / Gaussian splat PLYs with multi-order RANSAC + DBSCAN",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full segmentation pipeline")
    run.add_argument("--input", "-i", required=True, help="Input .ply / .pcd (or Gaussian training PLY)")
    run.add_argument("--out", "-o", required=True, help="Output directory")

    run.add_argument("--no-center", action="store_true", help="Do not translate cloud to origin")
    run.add_argument("--skip-outlier", action="store_true")
    run.add_argument("--outlier-nn", type=int, default=16)
    run.add_argument("--outlier-std", type=float, default=2.0)
    run.add_argument("--voxel", type=float, default=0.1, help="Voxel size for downsample (0 to skip)")
    run.add_argument("--skip-downsample", action="store_true", help="Keep all points (preserves Gaussian attrs)")
    run.add_argument("--skip-normals", action="store_true")

    run.add_argument("--max-planes", type=int, default=6)
    run.add_argument("--plane-dist", type=float, default=0.02, help="RANSAC point-to-plane distance")
    run.add_argument("--ransac-n", type=int, default=3)
    run.add_argument("--ransac-iters", type=int, default=1000)
    run.add_argument("--min-inliers", type=int, default=50)
    run.add_argument("--dbscan-eps", type=float, default=0.05)
    run.add_argument("--dbscan-min-points", type=int, default=5)
    run.add_argument("--skip-dbscan", action="store_true")
    run.add_argument("--keep-centered", action="store_true", help="Export in centered coordinates")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        voxel = 0.0 if args.skip_downsample else args.voxel
        pre = PreprocessConfig(
            center=not args.no_center,
            outlier_nn=args.outlier_nn,
            outlier_std=args.outlier_std,
            skip_outlier=args.skip_outlier,
            voxel_size=voxel,
            skip_downsample=args.skip_downsample or args.voxel <= 0,
            skip_normals=args.skip_normals,
        )
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
