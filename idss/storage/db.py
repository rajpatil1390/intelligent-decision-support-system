"""SQLite persistence with SQLAlchemy 2.0: datasets, runs and run events in workspace/idss.db.

The database is an index for listing and querying (the API reads it); each run's state.json remains the
source of truth for its full context. One engine is cached per database file, so pointing the workspace
somewhere else (as the tests do) transparently uses a different database.
"""

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import URL, DateTime, Engine, ForeignKey, String, Text, TypeDecorator, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from configs.settings import get_settings
from idss.core.logger import get_logger

logger = get_logger()

SQLITE_TIMEOUT_S = 30  # wait this long for a lock held by another thread before failing


class UTCDateTime(TypeDecorator[datetime]):
    """Stores naive UTC (SQLite has no time zones) and always returns timezone-aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    rows: Mapped[int]
    cols: Mapped[int]
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime)


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), index=True)
    question: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(20), index=True)
    task_type: Mapped[str | None] = mapped_column(String(40))
    best_model: Mapped[str | None] = mapped_column(String(64))
    primary_metric: Mapped[str | None] = mapped_column(String(40))
    metric_value: Mapped[float | None]
    verification_passed: Mapped[bool | None]
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    report_pdf: Mapped[str | None] = mapped_column(Text)
    report_md: Mapped[str | None] = mapped_column(Text)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime)
    from_state: Mapped[str | None] = mapped_column(String(20))
    to_state: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)


_engines: dict[Path, Engine] = {}
_engines_lock = threading.Lock()


def get_engine(db_path: Path | None = None) -> Engine:
    """The engine for `db_path` (default: settings.db_path), created with its tables on first use."""
    path = (db_path if db_path is not None else get_settings().db_path).resolve()
    with _engines_lock:
        engine = _engines.get(path)
        if engine is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(URL.create("sqlite", database=str(path)),
                                   connect_args={"check_same_thread": False, "timeout": SQLITE_TIMEOUT_S})
            event.listen(engine, "connect", _configure_connection)
            Base.metadata.create_all(engine)
            _engines[path] = engine
            logger.debug("Opened database {}.", path)
        return engine


def init_db(db_path: Path | None = None) -> Engine:
    """Create any missing tables (idempotent: existing tables and rows are left alone) and return the engine."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def _configure_connection(connection: Any, record: Any) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")  # readers (the API) never block the training thread's writes
    cursor.close()
