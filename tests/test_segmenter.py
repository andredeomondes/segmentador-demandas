"""Tests for the demand segmentation package."""

from __future__ import annotations

import pandas as pd
import pytest

from segmenter import (
    Dataset,
    PortugueseStemmer,
    SilhouetteSegmenter,
    load_dataset,
    vectorize,
)
from segmenter.vectorization import Vectorization


class PassThrough:
    """Minimal Preprocessor implementation used to isolate the tokenizer.

    Its presence also checks that the Preprocessor protocol accepts an
    implementation defined outside the package.
    """

    name = "passthrough"

    def __call__(self, document: str) -> str:
        return document


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset()


@pytest.fixture(scope="module")
def vectorization(dataset: Dataset) -> Vectorization:
    return vectorize(dataset.texts, PortugueseStemmer())


class TestVectorization:
    def test_matrix_has_one_row_per_document(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        assert vectorization.shape[0] == len(dataset)

    def test_terms_align_with_columns(self, vectorization: Vectorization) -> None:
        assert len(vectorization.terms) == vectorization.shape[1]

    def test_drops_terms_seen_in_a_single_document(self) -> None:
        texts = pd.Series(["cancelar plano", "cancelar pedido", "termo unico aqui"])
        result = vectorize(texts, PassThrough())
        assert "unico" not in result.terms


class TestSegmenter:
    def test_rejects_k_below_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            SilhouetteSegmenter(range(1, 5))

    def test_selects_k_from_the_evaluated_range(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        segmenter = SilhouetteSegmenter(range(2, 9))
        report = segmenter.segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert report.selected_k in range(2, 9)

    def test_selected_k_maximizes_silhouette(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        report = SilhouetteSegmenter(range(2, 9)).segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert report.selected_silhouette == max(report.silhouette_by_k.values())

    def test_assigns_every_document(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        report = SilhouetteSegmenter(range(2, 6)).segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert len(report.assignments) == len(dataset)

    def test_profiles_cover_every_group(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        report = SilhouetteSegmenter(range(2, 6)).segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert len(report.profiles) == report.selected_k
        assert sum(profile.size for profile in report.profiles) == len(dataset)

    def test_purity_is_a_proportion(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        report = SilhouetteSegmenter(range(2, 6)).segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert all(0 < profile.purity <= 1 for profile in report.profiles)

    def test_agreement_level_reflects_the_index(
        self, vectorization: Vectorization, dataset: Dataset
    ) -> None:
        report = SilhouetteSegmenter(range(2, 6)).segment(
            vectorization.matrix, vectorization.terms, dataset.labels
        )
        assert report.agreement_level in {"strong", "partial", "weak"}

    def test_identical_labels_and_groups_yield_perfect_agreement(self) -> None:
        texts = pd.Series(
            ["cancelar plano agora", "cancelar assinatura agora",
             "entrega atrasada demais", "entrega parada demais"]
        )
        labels = pd.Series(["cancelamento", "cancelamento", "entrega", "entrega"])
        result = vectorize(texts, PassThrough())
        report = SilhouetteSegmenter(range(2, 3)).segment(
            result.matrix, result.terms, labels
        )
        assert report.adjusted_rand == pytest.approx(1.0)
