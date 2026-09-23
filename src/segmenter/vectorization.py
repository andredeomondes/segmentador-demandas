"""Document-term representation for segmentation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import Preprocessor

_MIN_DOCUMENT_FREQUENCY = 2


@dataclasses.dataclass(frozen=True)
class Vectorization:
    """A document-term matrix and its feature names.

    Attributes:
        matrix: Sparse document-term matrix.
        terms: Feature names aligned with the matrix columns.
    """

    matrix: spmatrix
    terms: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Number of documents and number of terms."""
        return self.matrix.shape


def vectorize(texts: pd.Series, preprocessor: Preprocessor) -> Vectorization:
    """Builds the TF-IDF representation used for clustering.

    Terms appearing in a single document cannot help form a group, so they are
    discarded. Keeping them would only add sparse dimensions that push every
    document further apart.

    Args:
        texts: Documents to vectorize.
        preprocessor: Normalization applied before tokenization.

    Returns:
        A ``Vectorization`` holding the matrix and its feature names.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor,
        ngram_range=(1, 2),
        min_df=_MIN_DOCUMENT_FREQUENCY,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return Vectorization(matrix=matrix, terms=vectorizer.get_feature_names_out())
