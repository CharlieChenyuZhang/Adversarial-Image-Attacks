"""Tools for constructing bounded adversarial-image candidates."""

from .merge import MergeResult, merge_files, merge_images
from .vision import (
    DEFAULT_MODEL,
    OpenAIVisionEvaluator,
    VisionAnalysis,
    VisionComparison,
    VisionEvaluationError,
    image_to_data_url,
)

__all__ = [
    "DEFAULT_MODEL",
    "MergeResult",
    "OpenAIVisionEvaluator",
    "VisionAnalysis",
    "VisionComparison",
    "VisionEvaluationError",
    "image_to_data_url",
    "merge_files",
    "merge_images",
]
__version__ = "0.2.0"
