"""End-to-end segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .export import export_results
from .io import load_cloud
from .preprocess import PreprocessConfig, preprocess
from .segment import SegmentConfig, segment


@dataclass
class PipelineConfig:
    input_path: str | Path
    output_dir: str | Path
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    uncenter_export: bool = True


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    cloud = load_cloud(cfg.input_path)
    # Gaussian PLYs often need finer voxels; callers set params. Default path works for both.
    pre = preprocess(cloud, cfg.preprocess)
    if len(pre.pcd.points) == 0:
        raise RuntimeError("Preprocessing removed all points; relax outlier/voxel settings.")

    seg = segment(pre.pcd, cfg.segment)
    summary = export_results(
        Path(cfg.output_dir),
        pre.cloud,
        pre,
        seg,
        uncenter=cfg.uncenter_export,
    )
    summary["config"] = {
        "input": str(Path(cfg.input_path).resolve()),
        "output": str(Path(cfg.output_dir).resolve()),
        "preprocess": cfg.preprocess.__dict__,
        "segment": cfg.segment.__dict__,
    }
    return summary
