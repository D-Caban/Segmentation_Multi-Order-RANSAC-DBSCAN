"""Point-cloud and Gaussian-splat segmentation (RANSAC, graph objects, selectables)."""

from .pipeline import PipelineConfig, run_pipeline
from .selectables import SelectablesConfig, run_selectables

__all__ = ["PipelineConfig", "run_pipeline", "SelectablesConfig", "run_selectables"]
__version__ = "0.2.0"
