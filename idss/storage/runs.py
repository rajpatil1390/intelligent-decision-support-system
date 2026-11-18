"""Run identifiers and per-run artifact directories under workspace/runs."""

import re
import uuid
from datetime import datetime
from pathlib import Path

from configs.settings import get_settings

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def new_run_id() -> str:
    """A sortable, unique run id such as '20261005-143012-a1b2c3'."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str, runs_dir: Path | None = None, *, create: bool = True) -> Path:
    """Return the artifact directory for `run_id` (created by default).

    run_id may contain only letters, digits, '-' and '_', so it can never escape the runs directory.
    """
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run id {run_id!r}: use letters, digits, '-' or '_' (max 64 chars).")
    path = (runs_dir if runs_dir is not None else get_settings().runs_dir) / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
