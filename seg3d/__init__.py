"""Point-cloud and Gaussian-splat segmentation (multi-order RANSAC + DBSCAN)."""

from .pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
__version__ = "0.1.0"
