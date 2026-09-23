"""Command line entry point for the demand segmenter."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import reporting
from .clustering import SilhouetteSegmenter
from .dataset import load_dataset
from .preprocessing import PortugueseStemmer
from .vectorization import vectorize


def build_parser() -> argparse.ArgumentParser:
    """Builds the command line parser."""
    parser = argparse.ArgumentParser(
        prog="segmenter",
        description="Group support requests without using their labels.",
    )
    parser.add_argument("--corpus", type=Path, help="CSV file to load.")
    parser.add_argument("--min-k", type=int, default=2, help="Smallest number of groups.")
    parser.add_argument("--max-k", type=int, default=10, help="Largest number of groups.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Directory for generated figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the segmentation.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    dataset = load_dataset(args.corpus)
    print(
        f"Corpus: {len(dataset)} documents, "
        f"{len(dataset.class_names)} human categories"
    )
    print("labels are read only for the final comparison\n")

    vectorization = vectorize(dataset.texts, PortugueseStemmer())
    print(
        f"TF-IDF matrix: {vectorization.shape[0]} documents "
        f"x {vectorization.shape[1]} terms\n"
    )

    segmenter = SilhouetteSegmenter(range(args.min_k, args.max_k + 1))
    report = segmenter.segment(vectorization.matrix, vectorization.terms, dataset.labels)

    reporting.describe_selection(report)
    reporting.describe_profiles(report)
    reporting.describe_agreement(report)

    path = reporting.save_projection(vectorization, report, dataset.labels, args.output)
    print(f"\nFigure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
