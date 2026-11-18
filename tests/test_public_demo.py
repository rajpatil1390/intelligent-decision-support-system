"""Public demo mode (IDSS_PUBLIC_DEMO=1): the setting, the API restrictions, startup retention and the fast-only
re-plan. No network calls; the API runs on a temporary workspace with a fake LLM and a tiny budget."""

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import PUBLIC_DEMO_MAX_UPLOAD_BYTES, create_app
from configs.settings import Settings, get_settings
from idss.agents.manager import Orchestrator, RunState
from idss.core.schemas import Check, VerificationReport
from idss.storage.repository import EventRecord, RunRecord, get_repository
from idss.storage.retention import PROFILE_SUFFIX, PurgeResult, purge_older_than
from tests.test_api import TINY, assert_error, classification_df, fake_llm, upload, wait_for
from tests.test_step5 import CHURN_CSV, ONE_MODEL, churn_llm

QUESTION = "Which customers will say yes?"
OLD = timedelta(days=2)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh workspace (database, runs, uploads) for one test."""
    monkeypatch.setattr(get_settings(), "workspace_dir", tmp_path / "workspace")
    return tmp_path / "workspace"


def demo_client(**kwargs: object) -> TestClient:
    llm = fake_llm()
    return TestClient(create_app(llm_factory=lambda: llm, budget=TINY, public_demo=True, **kwargs))  # type: ignore[arg-type]


def set_mtime(path: Path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


def add_run(run_id: str, created_at: datetime, dataset_id: str | None = None) -> Path:
    """A finished run: database row + one event + a run directory with a file in it."""
    record = RunRecord(run_id=run_id, dataset_id=dataset_id, question=QUESTION, state="DONE",
                       created_at=created_at, updated_at=created_at)
    get_repository().record_transition(record, EventRecord(run_id=run_id, ts=created_at, to_state="DONE", message="ok"))
    directory = get_settings().runs_dir / run_id
    directory.mkdir(parents=True)
    (directory / "state.json").write_text("{}", encoding="utf-8")
    return directory


def add_dataset(dataset_id: str, uploaded_at: datetime) -> Path:
    uploads = get_settings().uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f"{dataset_id}_data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    Path(str(path) + PROFILE_SUFFIX).write_text("{}", encoding="utf-8")
    get_repository().add_dataset(dataset_id, "data.csv", path, 1, 2, uploaded_at=uploaded_at)
    return path


# --------------------------------------------------------------------------- setting


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("0", False), ("", False), ("no", False)])
def test_public_demo_setting_reads_env(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("IDSS_PUBLIC_DEMO", value)
    assert Settings(_env_file=None).public_demo is expected  # type: ignore[call-arg]


def test_public_demo_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDSS_PUBLIC_DEMO", raising=False)
    assert Settings(_env_file=None).public_demo is False  # type: ignore[call-arg]


def test_create_app_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert create_app().state.config.public_demo is False
    monkeypatch.setattr(get_settings(), "public_demo", True)
    config = create_app().state.config
    assert config.public_demo is True and config.max_upload_bytes == PUBLIC_DEMO_MAX_UPLOAD_BYTES
    assert create_app(public_demo=False).state.config.public_demo is False  # an explicit argument wins


# --------------------------------------------------------------------------- API restrictions


def test_health_reports_the_public_demo(workspace: Path) -> None:
    with demo_client() as client:
        body = client.get("/health").json()
    assert body["public_demo"] is True and body["max_upload_mb"] == 10.0
    with demo_client(max_upload_bytes=1_048_576) as client:  # a smaller explicit limit is kept
        assert client.get("/health").json()["max_upload_mb"] == 1.0


def test_public_demo_caps_uploads_at_10_mb(workspace: Path) -> None:
    with demo_client(max_upload_bytes=50 * 2**20) as client:
        too_big = b"a\n" + b"1\n" * (5 * 2**20 + 100_000)  # ~10.2 MB
        response = client.post("/datasets", files={"file": ("big.csv", too_big, "text/csv")})
        assert "10 MB" in assert_error(response, 413, "file_too_large")["detail"]
        upload(client, classification_df(50))  # small files are still accepted


def test_public_demo_runs_fast_only_and_disables_ask(workspace: Path) -> None:
    with demo_client() as client:
        dataset = upload(client, classification_df(300))
        body = {"dataset_id": dataset["id"], "question": QUESTION}
        detail = assert_error(client.post("/runs", json=body | {"mode": "full"}), 403, "public_demo")["detail"]
        assert "fast mode only" in detail
        response = client.post("/runs", json=body)  # mode defaults to fast
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        assert wait_for(client, run_id)["state"] == "DONE"
        detail = assert_error(client.post(f"/runs/{run_id}/ask", json={"question": "How many rows?"}),
                              403, "public_demo")["detail"]
        assert "disabled in the public demo" in detail
        # The rest of the results still work.
        assert client.get(f"/runs/{run_id}/leaderboard").status_code == 200


# --------------------------------------------------------------------------- retention


def test_purge_deletes_old_runs_and_unused_uploads(workspace: Path) -> None:
    now = datetime.now(UTC)
    old_unused = add_dataset("old_unused", now - OLD)
    old_used = add_dataset("old_used", now - OLD)
    recent = add_dataset("recent", now)
    old_run = add_run("old-run", now - OLD, "old_unused")
    new_run = add_run("new-run", now - timedelta(hours=1), "old_used")
    orphan_old, orphan_new = get_settings().runs_dir / "orphan-old", get_settings().runs_dir / "orphan-new"
    orphan_old.mkdir()
    orphan_new.mkdir()
    set_mtime(orphan_old, now - OLD)
    stray_old, stray_new = get_settings().uploads_dir / "stray_old.csv", get_settings().uploads_dir / "stray_new.csv"
    stray_old.write_text("x", encoding="utf-8")
    stray_new.write_text("x", encoding="utf-8")
    set_mtime(stray_old, now - OLD)

    result = purge_older_than(timedelta(hours=24), now=now)

    assert result == PurgeResult(runs=2, datasets=1, files=1)
    repo = get_repository()
    assert repo.get_run("old-run") is None and repo.list_events("old-run") == [] and not old_run.exists()
    assert repo.get_run("new-run") is not None and new_run.is_dir()
    assert not orphan_old.exists() and orphan_new.is_dir()
    assert repo.get_dataset("old_unused") is None and not old_unused.exists()
    assert not Path(str(old_unused) + PROFILE_SUFFIX).exists()
    assert repo.get_dataset("old_used") is not None and old_used.is_file()  # a newer run still uses it
    assert repo.get_dataset("recent") is not None and recent.is_file()
    assert Path(str(recent) + PROFILE_SUFFIX).is_file()  # profile caches of kept datasets are not strays
    assert not stray_old.exists() and stray_new.is_file()
    assert purge_older_than(timedelta(hours=24), now=now) == PurgeResult()  # nothing left to delete


def test_public_demo_startup_deletes_runs_older_than_24_hours(workspace: Path) -> None:
    now = datetime.now(UTC)
    old_run = add_run("old-run", now - timedelta(hours=25))
    new_run = add_run("new-run", now - timedelta(hours=23))
    with TestClient(create_app(llm_factory=lambda: None, public_demo=False)):
        pass  # the normal app keeps everything
    assert old_run.is_dir() and get_repository().get_run("old-run") is not None
    with demo_client() as client:
        assert [r["run_id"] for r in client.get("/runs").json()] == ["new-run"]
    assert not old_run.exists() and new_run.is_dir()


# --------------------------------------------------------------------------- orchestrator


def test_fast_only_keeps_the_verification_replan_in_fast_mode(tmp_path: Path) -> None:
    orchestrator = Orchestrator(churn_llm(), tmp_path, budget=ONE_MODEL, fast_only=True)
    failing = VerificationReport(passed=False, checks=[Check(name="metric_targets", passed=False, detail="too low")])
    orchestrator.verifier.verify = lambda *args, **kwargs: failing  # type: ignore[method-assign]
    run_id = orchestrator.start("Which customers will churn?", CHURN_CSV, budget_mode="full")
    deadline = time.monotonic() + 120
    ctx = orchestrator.context
    assert ctx is not None
    while not (ctx.plan_revision_count == 1 and ctx.state == RunState.TRAIN):
        assert time.monotonic() < deadline and ctx.state not in (RunState.DONE, RunState.FAILED), ctx.state
        orchestrator.step(run_id)
    plans = [event.message for event in ctx.history if event.state == RunState.TRAIN]
    assert len(plans) == 2 and all("fast mode" in message for message in plans), plans
    assert ctx.plan is not None and ctx.plan.budget_mode == "fast"
    assert any("re-planning once with the fast budget (public demo)" in e.message for e in ctx.history)
