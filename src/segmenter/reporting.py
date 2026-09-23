"""Console and figure output for segmentation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .clustering import RANDOM_STATE, SegmentationReport
from .vectorization import Vectorization

_FIGURE_DPI = 120
_OTHER_CATEGORIES_SHOWN = 3


def describe_selection(report: SegmentationReport) -> None:
    """Prints the silhouette score of every candidate number of groups."""
    print("Choosing the number of groups by silhouette")
    for k, score in report.silhouette_by_k.items():
        marker = " <- selected" if k == report.selected_k else ""
        print(f"  k={k:<3} {score:.3f}{marker}")


def describe_profiles(report: SegmentationReport) -> None:
    """Prints the defining terms and composition of each group."""
    print("\nDiscovered groups")
    for profile in report.profiles:
        print(f"\n  group {profile.label} ({profile.size} documents)")
        print(f"    terms      {', '.join(profile.top_terms)}")
        print(
            f"    dominant   {profile.dominant_category} "
            f"({100 * profile.purity:.0f}% of the group)"
        )
        others = profile.composition.iloc[1 : 1 + _OTHER_CATEGORIES_SHOWN]
        if not others.empty:
            listed = ", ".join(f"{name}={count}" for name, count in others.items())
            print(f"    also holds {listed}")


def describe_agreement(report: SegmentationReport) -> None:
    """Prints the comparison against the human categorization."""
    print("\nDiscovered groups against human categories")
    print(f"  adjusted Rand index  {report.adjusted_rand:.3f} ({report.agreement_level})")
    purest = report.purest_profile
    print(
        f"  purest group         {purest.label} -> {purest.dominant_category} "
        f"at {100 * purest.purity:.0f}%"
    )
    print("  a low index means the model split by a different criterion,")
    print("  not that the segmentation failed")


def save_projection(
    vectorization: Vectorization, report: SegmentationReport, labels: pd.Series,
    destination: Path,
) -> Path:
    """Writes a two-dimensional projection of groups beside human categories.

    Args:
        vectorization: Document-term matrix to project.
        report: Segmentation whose assignments are plotted.
        labels: Human categories, plotted for comparison.
        destination: Directory where the figure is written.

    Returns:
        Path of the written figure.
    """
    reducer = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    points = reducer.fit_transform(vectorization.matrix)
    explained = reducer.explained_variance_ratio_.sum()

    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 6))

    left.scatter(points[:, 0], points[:, 1], c=report.assignments, cmap="tab10",
                 s=38, alpha=0.85)
    left.set_title(f"Discovered groups (k={report.selected_k})")

    right.scatter(points[:, 0], points[:, 1], c=pd.Categorical(labels).codes,
                  cmap="tab10", s=38, alpha=0.85)
    right.set_title("Human categories")

    for axes in (left, right):
        axes.set_xlabel("component 1")
        axes.set_ylabel("component 2")

    figure.suptitle(f"SVD projection - {100 * explained:.1f}% of variance explained")
    figure.tight_layout()

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "segments.png"
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path
