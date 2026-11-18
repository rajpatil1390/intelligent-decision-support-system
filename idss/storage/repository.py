"""Simple CRUD over the datasets, runs and events tables.

Every method opens its own short session, so one Repository can be shared by the API's request threads
and the training thread. Results are Pydantic records, never live ORM objects.
"""

import threading
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from idss.storage.db import DatasetRow, EventRow, RunRow, get_engine

DEFAULT_LIMIT = 100


class _Record(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DatasetRecord(_Record):
    id: str
    filename: str
    path: str
    rows: int
    cols: int
    uploaded_at: datetime


class RunRecord(_Record):
    run_id: str
    dataset_id: str | None = None
    question: str
    state: str
    task_type: str | None = None
    best_model: str | None = None
    primary_metric: str | None = None
    metric_value: float | None = None
    verification_passed: bool | None = None
    created_at: datetime
    updated_at: datetime
    report_pdf: str | None = None
    report_md: str | None = None


class EventRecord(_Record):
    id: int | None = None
    run_id: str
    ts: datetime
    from_state: str | None = None
    to_state: str
    message: str


class Repository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._sessions.begin() as session:  # commits on success, rolls back on any exception
            yield session

    # ------------------------------------------------------------------ datasets

    def add_dataset(self, dataset_id: str, filename: str, path: str | Path, rows: int, cols: int,
                    uploaded_at: datetime | None = None) -> DatasetRecord:
        row = DatasetRow(id=dataset_id, filename=filename, path=str(path), rows=rows, cols=cols,
                         uploaded_at=uploaded_at or datetime.now(UTC))
        with self._session() as session:
            session.add(row)
        return DatasetRecord.model_validate(row)

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with self._session() as session:
            row = session.get(DatasetRow, dataset_id)
            return DatasetRecord.model_validate(row) if row is not None else None

    def list_datasets(self, limit: int | None = DEFAULT_LIMIT, offset: int = 0, *,
                      uploaded_before: datetime | None = None) -> list[DatasetRecord]:
        """Newest first (limit None = all), optionally only those uploaded before a cut-off."""
        query = select(DatasetRow).order_by(DatasetRow.uploaded_at.desc()).limit(limit).offset(offset)
        if uploaded_before is not None:
            query = query.where(DatasetRow.uploaded_at < uploaded_before)
        with self._session() as session:
            return [DatasetRecord.model_validate(row) for row in session.scalars(query)]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete the dataset row (runs keep their history; their dataset_id becomes NULL)."""
        with self._session() as session:
            return session.execute(delete(DatasetRow).where(DatasetRow.id == dataset_id)).rowcount > 0

    # ------------------------------------------------------------------ runs

    def save_run(self, run: RunRecord) -> RunRecord:
        """Insert the run, or overwrite every column of an existing one."""
        with self._session() as session:
            session.merge(RunRow(**run.model_dump()))
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._session() as session:
            row = session.get(RunRow, run_id)
            return RunRecord.model_validate(row) if row is not None else None

    def list_runs(self, limit: int | None = DEFAULT_LIMIT, offset: int = 0, *, dataset_id: str | None = None,
                  states: Collection[str] | None = None, created_before: datetime | None = None) -> list[RunRecord]:
        """Newest first (limit None = all), optionally filtered by dataset, a set of states and/or a cut-off."""
        query = select(RunRow).order_by(RunRow.created_at.desc()).limit(limit).offset(offset)
        if created_before is not None:
            query = query.where(RunRow.created_at < created_before)
        if dataset_id is not None:
            query = query.where(RunRow.dataset_id == dataset_id)
        if states is not None:
            query = query.where(RunRow.state.in_(list(states)))
        with self._session() as session:
            return [RunRecord.model_validate(row) for row in session.scalars(query)]

    def delete_run(self, run_id: str) -> bool:
        """Delete the run and its events."""
        with self._session() as session:
            session.execute(delete(EventRow).where(EventRow.run_id == run_id))
            return session.execute(delete(RunRow).where(RunRow.run_id == run_id)).rowcount > 0

    # ------------------------------------------------------------------ events

    def add_event(self, event: EventRecord) -> EventRecord:
        row = EventRow(**event.model_dump(exclude={"id"}))
        with self._session() as session:
            session.add(row)
        return EventRecord.model_validate(row)

    def list_events(self, run_id: str) -> list[EventRecord]:
        query = select(EventRow).where(EventRow.run_id == run_id).order_by(EventRow.id)
        with self._session() as session:
            return [EventRecord.model_validate(row) for row in session.scalars(query)]

    def record_transition(self, run: RunRecord, event: EventRecord | None = None) -> None:
        """Save the run's summary row and (optionally) its new event in one transaction."""
        with self._session() as session:
            session.merge(RunRow(**run.model_dump()))
            if event is not None:
                session.flush()  # the run row must exist before its event (foreign key)
                session.add(EventRow(**event.model_dump(exclude={"id"})))


_repositories: dict[int, Repository] = {}
_repositories_lock = threading.Lock()


def get_repository(db_path: Path | None = None) -> Repository:
    """The shared Repository for `db_path` (default: settings.db_path); tables are created on first use."""
    engine = get_engine(db_path)
    with _repositories_lock:
        repository = _repositories.get(id(engine))
        if repository is None:
            repository = _repositories[id(engine)] = Repository(engine)
        return repository
