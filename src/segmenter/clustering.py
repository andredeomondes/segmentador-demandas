"""Unsupervised segmentation of support requests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

RANDOM_STATE = 42

_DEFAULT_K_RANGE = range(2, 11)
_TOP_TERMS = 6

_STRONG_AGREEMENT = 0.5
_PARTIAL_AGREEMENT = 0.2


@dataclasses.dataclass(frozen=True)
class ClusterProfile:
    """Description of one discovered group.

    Attributes:
        label: Identifier assigned by the clustering algorithm.
        size: Number of documents in the group.
        top_terms: Terms with the highest weight at the group centroid.
        dominant_category: Most frequent human category inside the group.
        purity: Share of the group holding the dominant category.
        composition: Count of every human category present.
    """

    label: int
    size: int
    top_terms: tuple[str, ...]
    dominant_category: str
    purity: float
    composition: pd.Series


@dataclasses.dataclass(frozen=True)
class SegmentationReport:
    """Outcome of segmenting a corpus without using its labels.

    Attributes:
        assignments: Group assigned to each document.
        selected_k: Number of groups chosen by the silhouette criterion.
        silhouette_by_k: Silhouette score for every candidate ``k``.
        profiles: Description of each discovered group.
        adjusted_rand: Agreement with the human categorization.
    """

    assignments: np.ndarray
    selected_k: int
    silhouette_by_k: dict[int, float]
    profiles: tuple[ClusterProfile, ...]
    adjusted_rand: float

    @property
    def selected_silhouette(self) -> float:
        """Silhouette score at the chosen ``k``."""
        return self.silhouette_by_k[self.selected_k]

    @property
    def agreement_level(self) -> str:
        """Qualitative reading of the adjusted Rand index."""
        if self.adjusted_rand >= _STRONG_AGREEMENT:
            return "strong"
        if self.adjusted_rand >= _PARTIAL_AGREEMENT:
            return "partial"
        return "weak"

    @property
    def purest_profile(self) -> ClusterProfile:
        """Group that best matches a single human category."""
        return max(self.profiles, key=lambda profile: profile.purity)


class SilhouetteSegmenter:
    """Groups documents and picks the number of groups from the data alone.

    In a real segmentation task there is no ground truth stating how many
    groups exist. The silhouette score measures cohesion within groups against
    separation between them using only the geometry of the vectors, which makes
    it an honest selection criterion. The human labels are read once, at the
    end, purely to quantify the disagreement.
    """

    def __init__(self, k_range: range = _DEFAULT_K_RANGE) -> None:
        """Initializes the segmenter.

        Args:
            k_range: Candidate numbers of groups to evaluate.

        Raises:
            ValueError: If ``k_range`` contains a value below two.
        """
        if min(k_range) < 2:
            raise ValueError(f"k must be at least 2, got {min(k_range)}")
        self._k_range = k_range

    def segment(
        self, matrix: spmatrix, terms: np.ndarray, labels: pd.Series
    ) -> SegmentationReport:
        """Segments the corpus and profiles each discovered group.

        Args:
            matrix: Document-term matrix.
            terms: Feature names aligned with the columns of ``matrix``.
            labels: Human categories, used only for the final comparison.

        Returns:
            A ``SegmentationReport`` describing the segmentation.
        """
        silhouette_by_k = {k: self._silhouette(matrix, k) for k in self._k_range}
        selected_k = max(silhouette_by_k, key=silhouette_by_k.get)

        model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
        assignments = model.fit_predict(matrix)

        return SegmentationReport(
            assignments=assignments,
            selected_k=selected_k,
            silhouette_by_k=silhouette_by_k,
            profiles=self._profile_all(model, terms, assignments, labels),
            adjusted_rand=float(adjusted_rand_score(labels, assignments)),
        )

    @staticmethod
    def _silhouette(matrix: spmatrix, k: int) -> float:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        return float(silhouette_score(matrix, model.fit_predict(matrix)))

    @staticmethod
    def _profile_all(
        model: KMeans, terms: np.ndarray, assignments: np.ndarray, labels: pd.Series
    ) -> tuple[ClusterProfile, ...]:
        profiles = []
        for label in sorted(set(assignments)):
            centroid = model.cluster_centers_[label]
            inside = labels[assignments == label]
            composition = inside.value_counts()
            profiles.append(
                ClusterProfile(
                    label=int(label),
                    size=len(inside),
                    top_terms=tuple(terms[centroid.argsort()[::-1][:_TOP_TERMS]]),
                    dominant_category=str(composition.index[0]),
                    purity=float(composition.iloc[0] / len(inside)),
                    composition=composition,
                )
            )
        return tuple(profiles)
