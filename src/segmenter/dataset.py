"""Dataset loading for the support request corpus."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

_DEFAULT_CORPUS = Path(__file__).parent / "data" / "tickets.csv"

_TEXT_COLUMN = "text"
_LABEL_COLUMN = "category"


@dataclasses.dataclass(frozen=True)
class Dataset:
    """An immutable corpus of documents and their human categories.

    The categories are carried alongside the texts but take no part in the
    segmentation. They are read once, at the end, to quantify how far the
    discovered groups fall from the existing taxonomy.

    Attributes:
        texts: Raw request texts, one per document.
        labels: Category assigned to each document, aligned with ``texts``.
    """

    texts: pd.Series
    labels: pd.Series

    def __len__(self) -> int:
        """Returns the number of documents."""
        return len(self.texts)

    @property
    def class_names(self) -> list[str]:
        """Sorted unique category names."""
        return sorted(self.labels.unique())


def load_dataset(path: Path | str | None = None) -> Dataset:
    """Loads the support request corpus from a CSV file.

    Args:
        path: Location of the CSV file. Defaults to the corpus bundled with
            the package.

    Returns:
        A ``Dataset`` holding the texts and their categories.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the required columns are missing or the file is empty.
    """
    source = Path(path) if path is not None else _DEFAULT_CORPUS
    if not source.is_file():
        raise FileNotFoundError(f"Corpus not found: {source}")

    frame = pd.read_csv(source)

    missing = {_TEXT_COLUMN, _LABEL_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Corpus is empty: {source}")

    return Dataset(texts=frame[_TEXT_COLUMN], labels=frame[_LABEL_COLUMN])
