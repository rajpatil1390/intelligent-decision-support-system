"""SQLite persistence (Step 7A): repository CRUD, idempotent init_db and the Orchestrator's mirroring."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from idss.agents.manager import Orchestrator, RunState
from idss.agents.requirement_agent import RelevanceCheck
from idss.storage.db import init_db
from idss.storage.repository import EventRecord, Repository, RunRecord, get_repository
from tests.fakes import FakeLLM

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return get_repository(tmp_path / "test.db")


def run(run_id: str, state: str = "INIT", minutes: int = 0, **extra: Any) -> RunRecord:
    when = T0 + timedelta(minutes=minutes)
    return RunRecord(run_id=run_id, question="Q?", state=state, created_at=when, updated_at=when, **extra)


def test_init_db_creates_tables_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "idss.db"
    engine = init_db(path)
    assert set(inspect(engine).get_table_names()) == {"datasets", "runs", "events"}
    get_repository(path).add_dataset("d1", "a.csv", tmp_path / "a.csv", 10, 3)
    assert init_db(path) is engine and init_db(path) is engine  # same engine, nothing dropped
    assert get_repository(path).get_dataset("d1") is not None


def test_dataset_crud(repo: Repository, tmp_path: Path) -> None:
    first = repo.add_dataset("d1", "a.csv", tmp_path / "a.csv", 10, 3, uploaded_at=T0)
    repo.add_dataset("d2", "b.xlsx", tmp_path / "b.xlsx", 20, 4, uploaded_at=T0 + timedelta(hours=1))
    assert first.path == str(tmp_path / "a.csv") and first.uploaded_at == T0
    assert repo.get_dataset("d1") == first and repo.get_dataset("missing") is None
    assert [d.id for d in repo.list_datasets()] == ["d2", "d1"]  # newest first
    assert [d.id for d in repo.list_datasets(limit=1, offset=1)] == ["d1"]
    assert repo.delete_dataset("d1") and not repo.delete_dataset("d1")
    assert [d.id for d in repo.list_datasets()] == ["d2"]


def test_run_and_event_crud(repo: Repository) -> None:
    repo.add_dataset("d1", "a.csv", "a.csv", 10, 3)
    repo.save_run(run("r1", dataset_id="d1"))
    repo.save_run(run("r2", minutes=5))
    repo.save_run(run("r1", state="DONE", dataset_id="d1", best_model="ridge", metric_value=0.9, verification_passed=True))
    saved = repo.get_run("r1")
    assert saved.state == "DONE" and saved.best_model == "ridge" and saved.metric_value == 0.9 and saved.created_at == T0
    assert [r.run_id for r in repo.list_runs()] == ["r2", "r1"]
    assert [r.run_id for r in repo.list_runs(dataset_id="d1")] == ["r1"]
    assert [r.run_id for r in repo.list_runs(states=["INIT", "TRAIN"])] == ["r2"]

    first = repo.add_event(EventRecord(run_id="r1", ts=T0, to_state="INIT", message="created"))
    repo.record_transition(run("r1", state="PARSE"), EventRecord(run_id="r1", ts=T0, from_state="INIT", to_state="PARSE",
                                                                  message="parsed"))
    events = repo.list_events("r1")
    assert first.id is not None and [(e.from_state, e.to_state) for e in events] == [(None, "INIT"), ("INIT", "PARSE")]
    assert repo.get_run("r1").state == "PARSE"

    assert repo.delete_dataset("d1") and repo.get_run("r1").dataset_id is None  # runs outlive their dataset
    assert repo.delete_run("r1") and repo.get_run("r1") is None and repo.list_events("r1") == []


def test_event_needs_an_existing_run(repo: Repository) -> None:
    with pytest.raises(IntegrityError):  # foreign keys are enforced
        repo.add_event(EventRecord(run_id="ghost", ts=T0, to_state="INIT", message="x"))


def test_orchestrator_mirrors_every_transition(repo: Repository, tmp_path: Path) -> None:
    orchestrator = Orchestrator(FakeLLM({RelevanceCheck: RelevanceCheck(relevant=False)}), tmp_path / "runs",
                                repository=repo)
    run_id = orchestrator.start("hello", tmp_path / "data.csv")
    ctx = orchestrator.run_until_blocked()
    assert ctx.state == RunState.DONE
    record = repo.get_run(run_id)
    assert record.state == "DONE" and record.question == "hello" and record.updated_at == ctx.updated_at
    assert [(e.from_state, e.to_state) for e in repo.list_events(run_id)] == [(None, "INIT"), ("INIT", "DONE")]


class BrokenRepository:
    def record_transition(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is locked")


def test_database_failure_never_stops_a_run(tmp_path: Path) -> None:
    orchestrator = Orchestrator(FakeLLM({RelevanceCheck: RelevanceCheck(relevant=False)}), tmp_path / "runs",
                                repository=BrokenRepository())  # type: ignore[arg-type]
    orchestrator.start("hello", tmp_path / "data.csv")
    assert orchestrator.run_until_blocked().state == RunState.DONE  # state.json is the source of truth
