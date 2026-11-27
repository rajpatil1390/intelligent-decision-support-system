"""Load tabular datasets (CSV, Excel, JSON, Parquet) into a pandas DataFrame.

Malformed files end in a DataLoadError with a plain explanation, never in an unexpected exception: empty files,
a header without rows, duplicate column names and unreadable bytes are refused. Recoverable oddities (a file
that is not UTF-8 or mixes encodings, more than WIDE_DATASET_COLUMNS columns) load with a warning in
`df.attrs["load_warnings"]`, which the profiler adds to the dataset's warnings.
"""

import codecs
import csv
import json
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from idss.core.logger import get_logger

logger = get_logger()

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json", ".parquet")
LARGE_DATASET_ROWS = 200_000
WIDE_DATASET_COLUMNS = 1_000
LOAD_WARNINGS_ATTR = "load_warnings"
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # utf-8-sig strips a BOM; latin-1 decodes any byte sequence
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_SNIFF_BYTES = 64 * 1024
_DECODE_CHUNK_BYTES = 1024 * 1024
_CSV_DELIMITERS = ",;\t|"


class DataLoadError(Exception):
    """A dataset file is missing, unsupported, empty, malformed or unreadable."""


def load_dataset(path: str | Path, *, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load a dataset file into a DataFrame.

    Args:
        path: .csv, .xlsx, .json or .parquet file.
        max_rows: if set and the file has more rows, keep a random sample of this many rows
            (original row order is preserved).
        seed: random seed for sampling.

    Raises:
        DataLoadError: for missing, unsupported, empty, malformed or corrupt files.
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

    warnings: list[str] = []
    try:
        df = reader(path, warnings)
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"'{path.name}' is empty: it has no header and no data.") from None
    except Exception as exc:  # pandas/openpyxl/pyarrow raise many unrelated types for corrupt files
        raise DataLoadError(f"Could not read '{path.name}' as {suffix[1:].upper()}: {type(exc).__name__}: {exc}") from exc

    if df.shape[1] == 0:
        raise DataLoadError(f"'{path.name}' has no columns.")
    if df.empty:
        raise DataLoadError(f"'{path.name}' has a header but no data rows.")
    df.columns = [str(col).strip() for col in df.columns]
    _reject_duplicate_names(list(df.columns), path)

    n_rows = len(df)
    if n_rows > LARGE_DATASET_ROWS:
        logger.warning(
            "'{}' has {:,} rows (more than {:,}); profiling and training may be slow. Pass max_rows to sample.",
            path.name, n_rows, LARGE_DATASET_ROWS,
        )
    if df.shape[1] > WIDE_DATASET_COLUMNS:
        warnings.append(f"The file has {df.shape[1]:,} columns (more than {WIDE_DATASET_COLUMNS:,}); profiling and "
                        "training will be slow. Consider keeping only the relevant columns.")
    if max_rows is not None and n_rows > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
        logger.info("Sampled {:,} of {:,} rows from '{}'.", max_rows, n_rows, path.name)

    for warning in warnings:
        logger.warning("'{}': {}", path.name, warning)
    df.attrs[LOAD_WARNINGS_ATTR] = warnings
    logger.info("Loaded '{}': {:,} rows x {} columns.", path.name, len(df), df.shape[1])
    return df


def load_warnings(df: pd.DataFrame) -> list[str]:
    """Warnings recorded by load_dataset for `df` (empty for frames that did not come from it)."""
    return list(df.attrs.get(LOAD_WARNINGS_ATTR, []))


def _reject_duplicate_names(names: list[str], path: Path) -> None:
    duplicates = [name for name, count in Counter(names).items() if count > 1 and name]
    if duplicates:
        shown = ", ".join(f"'{name}'" for name in duplicates[:5])
        more = f" and {len(duplicates) - 5} more" if len(duplicates) > 5 else ""
        raise DataLoadError(f"'{path.name}' has duplicate column names: {shown}{more}. Give every column a unique name.")


def _read_csv(path: Path, warnings: list[str]) -> pd.DataFrame:
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    utf16 = head.startswith(_UTF16_BOMS)
    if b"\x00" in head and not utf16:
        raise DataLoadError(f"'{path.name}' looks like a binary file, not CSV text.")
    if not head.strip():
        raise DataLoadError(f"'{path.name}' is empty: it has no header and no data.")

    sample = head.decode("utf-16" if utf16 else "utf-8", errors="replace")
    if "\n" in sample:
        sample = sample[: sample.rfind("\n")]  # sniff complete lines only
    delimiter = _sniff_delimiter(sample)

    encoding = "utf-16" if utf16 else _choose_encoding(path, warnings)
    # pandas renames repeated header names ('a', 'a.1'), so the raw header row is checked first.
    header = pd.read_csv(path, sep=delimiter, encoding=encoding, header=None, nrows=1, dtype=str, keep_default_na=False)
    _reject_duplicate_names([str(name).strip() for name in header.iloc[0].tolist()] if len(header) else [], path)
    df = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
    logger.debug("Read '{}' with encoding={} delimiter={!r}.", path.name, encoding, delimiter)
    return df


def _choose_encoding(path: Path, warnings: list[str]) -> str:
    """UTF-8 if the bytes decode as UTF-8, else Windows-1252, else latin-1 (with a warning either way)."""
    if _decodes(path, "utf-8"):
        return CSV_ENCODINGS[0]
    for encoding in CSV_ENCODINGS[1:]:
        if not _decodes(path, encoding):
            continue
        label = "Windows-1252" if encoding == "cp1252" else "Latin-1"
        detail = (" It mixes UTF-8 with another encoding, so some characters may look wrong; re-save the file as "
                  "UTF-8." if _has_utf8_characters(path) else "")
        warnings.append(f"The file is not UTF-8; it was read as {label}.{detail}")
        return encoding
    raise DataLoadError(f"Could not decode '{path.name}' with any of: {', '.join(CSV_ENCODINGS)}.")  # unreachable


def _chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(_DECODE_CHUNK_BYTES):
            yield chunk


def _decodes(path: Path, encoding: str) -> bool:
    """Whether the whole file decodes strictly as `encoding` (read in chunks, so memory stays bounded)."""
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        for chunk in _chunks(path):
            decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _has_utf8_characters(path: Path) -> bool:
    """Whether a file that is not valid UTF-8 still contains valid multi-byte UTF-8 characters (mixed encodings)."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return any(any(ord(char) > 127 and char != "�" for char in decoder.decode(chunk)) for chunk in _chunks(path))


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","  # single-column files have no detectable delimiter


def _read_excel(path: Path, warnings: list[str]) -> pd.DataFrame:
    header = pd.read_excel(path, engine="openpyxl", header=None, nrows=1)
    _reject_duplicate_names([str(name).strip() for name in header.iloc[0].dropna().tolist()] if len(header) else [], path)
    return pd.read_excel(path, engine="openpyxl")


def _read_json(path: Path, warnings: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, lines=True)  # JSON Lines: one record per line
    # Nested lists/dicts are unhashable and break profiling; keep them as JSON strings.
    for col in df.columns:
        if df[col].map(lambda value: isinstance(value, list | dict)).any():
            df[col] = df[col].map(lambda value: json.dumps(value) if isinstance(value, list | dict) else value)
    return df


def _read_parquet(path: Path, warnings: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path)


_READERS: dict[str, Callable[[Path, list[str]], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".xlsx": _read_excel,
    ".json": _read_json,
    ".parquet": _read_parquet,
}
