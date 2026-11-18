"""Load tabular datasets (CSV, Excel, JSON, Parquet) into a pandas DataFrame."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from idss.core.logger import get_logger

logger = get_logger()

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json", ".parquet")
LARGE_DATASET_ROWS = 200_000
CSV_ENCODINGS = ("utf-8-sig", "latin-1")  # utf-8-sig also strips a BOM; latin-1 decodes any byte sequence
_SNIFF_BYTES = 64 * 1024
_CSV_DELIMITERS = ",;\t|"


class DataLoadError(Exception):
    """A dataset file is missing, unsupported, empty or unreadable."""


def load_dataset(path: str | Path, *, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load a dataset file into a DataFrame.

    Args:
        path: .csv, .xlsx, .json or .parquet file.
        max_rows: if set and the file has more rows, keep a random sample of this many rows
            (original row order is preserved).
        seed: random seed for sampling.

    Raises:
        DataLoadError: for missing, unsupported, empty or corrupt files.
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer.")
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"File not found: {path}")
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise DataLoadError(f"Unsupported file type '{suffix or path.name}'. Supported types: {supported}.")
    if path.stat().st_size == 0:
        raise DataLoadError(f"File is empty: {path.name}")

    try:
        df = reader(path)
    except DataLoadError:
        raise
    except Exception as exc:  # pandas/openpyxl/pyarrow raise many unrelated types for corrupt files
        raise DataLoadError(f"Could not read '{path.name}' as {suffix[1:].upper()}: {type(exc).__name__}: {exc}") from exc

    if df.shape[1] == 0 or df.empty:
        raise DataLoadError(f"'{path.name}' has no data rows.")
    df.columns = [str(col).strip() for col in df.columns]

    n_rows = len(df)
    if n_rows > LARGE_DATASET_ROWS:
        logger.warning(
            "'{}' has {:,} rows (more than {:,}); profiling and training may be slow. Pass max_rows to sample.",
            path.name, n_rows, LARGE_DATASET_ROWS,
        )
    if max_rows is not None and n_rows > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
        logger.info("Sampled {:,} of {:,} rows from '{}'.", max_rows, n_rows, path.name)

    logger.info("Loaded '{}': {:,} rows x {} columns.", path.name, len(df), df.shape[1])
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    if b"\x00" in head:
        raise DataLoadError(f"'{path.name}' looks like a binary file, not CSV text.")

    sample = head.decode("utf-8", errors="replace")
    if "\n" in sample:
        sample = sample[: sample.rfind("\n")]  # sniff complete lines only
    delimiter = _sniff_delimiter(sample)

    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            logger.debug("'{}' is not valid {}; trying the next encoding.", path.name, encoding)
            continue
        logger.debug("Read '{}' with encoding={} delimiter={!r}.", path.name, encoding, delimiter)
        return df
    raise DataLoadError(f"Could not decode '{path.name}' with any of: {', '.join(CSV_ENCODINGS)}.")


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","  # single-column files have no detectable delimiter


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _read_json(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, lines=True)  # JSON Lines: one record per line
    # Nested lists/dicts are unhashable and break profiling; keep them as JSON strings.
    for col in df.columns:
        if df[col].map(lambda value: isinstance(value, list | dict)).any():
            df[col] = df[col].map(lambda value: json.dumps(value) if isinstance(value, list | dict) else value)
    return df


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".xlsx": _read_excel,
    ".json": _read_json,
    ".parquet": _read_parquet,
}
