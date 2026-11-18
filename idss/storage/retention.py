"""Retention for the public demo: delete runs older than a cut-off, and the uploads no remaining run needs.

Called once at API startup when IDSS_PUBLIC_DEMO is on (api/main.py), so visitors' data does not pile up.
Deleted are, for every item older than the cut-off:
- runs in the database: the run directory (state.json, models, charts, reports), the row and its events;
- run directories that have no database row (e.g. left by a crash), judged by the directory's modification time;
- datasets that no remaining run uses: the uploaded file, its cached profile and the row;
- stray files in the uploads folder that no dataset row points to.
A dataset uploaded earlier but used by a newer run is kept until that run expires too.
"""

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from configs.settings import get_settings
from idss.core.logger import get_logger
from idss.storage.repository import Repository, get_repository
from idss.storage.runs import run_dir

logger = get_logger()

PROFILE_SUFFIX = ".profile.json"  # cached profile next to an uploaded file (written by the API)


@dataclass
class PurgeResult:
    runs: int = 0
    datasets: int = 0
    files: int = 0


def _modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _remove_tree(path: Path) -> bool:
    """Delete a directory; False (and a warning) if it could not be removed, e.g. a file open on Windows."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning("Could not delete {}: {}", path, exc)
        return False
    return True


def _remove_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete {}: {}", path, exc)
        return False
    return True


def purge_older_than(max_age: timedelta, *, now: datetime | None = None,
                     repository: Repository | None = None) -> PurgeResult:
    """Delete runs created, and datasets uploaded, more than `max_age` before `now` (see the module docstring)."""
    settings = get_settings()
    repo = repository or get_repository()
    cutoff = (now or datetime.now(UTC)) - max_age
    result = PurgeResult()

    for record in repo.list_runs(limit=None, created_before=cutoff):
        try:
            directory = run_dir(record.run_id, create=False)
        except ValueError:  # not a valid run id: there is no directory to delete
            directory = None
        if directory is None or _remove_tree(directory):  # keep the row if the files are still there
            repo.delete_run(record.run_id)
            result.runs += 1

    if settings.runs_dir.is_dir():
        for directory in settings.runs_dir.iterdir():
            if (directory.is_dir() and repo.get_run(directory.name) is None and _modified(directory) < cutoff
                    and _remove_tree(directory)):
                result.runs += 1

    for dataset in repo.list_datasets(limit=None, uploaded_before=cutoff):
        if repo.list_runs(limit=1, dataset_id=dataset.id):
            continue  # a newer run still needs this file (what-if, reports)
        path = Path(dataset.path)
        if _remove_file(path) and _remove_file(Path(str(path) + PROFILE_SUFFIX)):
            repo.delete_dataset(dataset.id)
            result.datasets += 1

    if settings.uploads_dir.is_dir():
        known = {Path(d.path).resolve() for d in repo.list_datasets(limit=None)}
        for path in settings.uploads_dir.iterdir():
            source = Path(str(path).removesuffix(PROFILE_SUFFIX)).resolve()
            if path.is_file() and source not in known and _modified(path) < cutoff and _remove_file(path):
                result.files += 1

    if result.runs or result.datasets or result.files:
        logger.info("Public demo retention: deleted {} run(s), {} dataset(s) and {} stray upload file(s) older than {}.",
                    result.runs, result.datasets, result.files, max_age)
    return result
