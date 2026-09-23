"""Unsupervised segmentation of support requests."""

from .clustering import ClusterProfile, SegmentationReport, SilhouetteSegmenter
from .dataset import Dataset, load_dataset
from .preprocessing import PortugueseStemmer, Preprocessor
from .vectorization import Vectorization, vectorize

__all__ = [
    "ClusterProfile",
    "Dataset",
    "PortugueseStemmer",
    "Preprocessor",
    "SegmentationReport",
    "SilhouetteSegmenter",
    "Vectorization",
    "load_dataset",
    "vectorize",
]
