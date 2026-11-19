"""IDSS API (FastAPI): upload datasets, start and follow runs, and use their results.

- Runs execute on one background worker thread (api/worker.py); a second training request while one is
  active gets 409, which keeps a laptop responsive.
- Each run's state.json is the source of truth for its details; the SQLite database (idss.storage) indexes
  datasets, runs and events for the list endpoints.
- Every error is JSON {"error": <code>, "detail": <message>}. Responses never contain secrets or server paths.
- Public demo (IDSS_PUBLIC_DEMO=1): Ask Your Data is off, uploads are capped at 10 MB, runs use the fast budget
  only, SHAP explains at most 200 rows (less memory), and runs older than 24 h are deleted at startup; GET /health
  tells the UI so it can show a banner.
- Startup creates the workspace folders if they are missing (hosted disks are ephemeral).
Start locally with scripts/run_api.ps1; interactive docs at http://localhost:8000/docs. In embedded mode
(IDSS_EMBEDDED_API=1) the Streamlit UI starts this app itself (ui/embedded_api.py).
"""

import re
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from api.schemas import (
    AnswerRequest,
    AskRequest,
    DatasetOut,
    DatasetUploaded,
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    LeaderboardResponse,
    ProfileSummary,
    RecommendationsResponse,
    RunCreate,
    RunCreated,
    RunDetail,
    RunEventOut,
    RunSummary,
    WhatIfRequest,
)
from api.worker import RunWorker, WorkerBusy
from configs.settings import get_settings
from idss.agents.analysis_agent import AnalysisAnswer
from idss.agents.manager import BLOCKED_STATES, Orchestrator, RunContext, RunState, load_context, run_record
from idss.agents.requirement_agent import requirement_from_profile
from idss.core.llm_client import LLMClient, LLMError
from idss.core.logger import get_logger
from idss.core.schemas import DatasetProfile, DecisionInsight, SegmentImpact, TrainingBudget, WhatIfResult
from idss.decision.what_if import WhatIfError
from idss.ml.data_loader import SUPPORTED_EXTENSIONS, DataLoadError, load_dataset
from idss.ml.profiler import profile_dataset
from idss.storage.db import init_db
from idss.storage.repository import DatasetRecord, RunRecord, get_repository
from idss.storage.retention import PROFILE_SUFFIX, purge_older_than
from idss.storage.runs import ensure_workspace, run_dir

logger = get_logger()

API_TITLE = "IDSS API"
API_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PUBLIC_DEMO_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PUBLIC_DEMO_RETENTION = timedelta(hours=24)  # older runs (and uploads no run needs) are deleted at startup
PUBLIC_DEMO_MAX_EXPLAIN_ROWS = 200  # SHAP sample in the public demo (default 500), to keep memory low
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form boundaries and headers around the file in a multipart body
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FILENAME_STEM = 80
CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]  # the Streamlit UI
LIST_LIMIT = 500
RESTART_REASON = "The server restarted while this run was in progress. Please start a new run."
# Share of the pipeline finished when a run is in each state (terminal states count as finished).
STATE_PROGRESS: dict[str, int] = {
    "INIT": 0, "PARSE": 5, "VALIDATE": 10, "NEEDS_INPUT": 10, "PROFILE": 15, "PLAN": 25, "TRAIN": 30,
    "VERIFY": 70, "EXPLAIN": 75, "INSIGHT": 90, "REPORT": 95, "DONE": 100, "FAILED": 100,
}
_CHART_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}(?:\.png)?")
_ERROR_CODES = {400: "bad_request", 403: "forbidden", 404: "not_found", 405: "method_not_allowed", 409: "conflict",
                413: "file_too_large", 415: "unsupported_file_type", 422: "validation_error",
                500: "internal_error", 503: "unavailable"}


class ApiError(Exception):
    """An error returned to the client as {error, detail} with the given HTTP status."""

    def __init__(self, status_code: int, error: str, detail: Any) -> None:
        super().__init__(f"{status_code} {error}: {detail}")
        self.status_code = status_code
        self.error = error
        self.detail = detail


@dataclass
class ApiConfig:
    llm_factory: Callable[[], LLMClient | None]
    budget: TrainingBudget | None = None  # overrides every run's training budget (tests use a tiny one)
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    public_demo: bool = False


def default_llm_factory() -> LLMClient | None:
    """The configured LLM client, or None (runs then work offline with deterministic fallbacks)."""
    settings = get_settings()
    if not settings.llm_configured:
        return None
    try:
        return LLMClient(settings)
    except LLMError as exc:
        logger.warning("LLM client unavailable ({}); continuing without an LLM.", exc)
        return None


# --------------------------------------------------------------------------- dependencies and helpers


def get_config(request: Request) -> ApiConfig:
    return request.app.state.config


def get_worker(request: Request) -> RunWorker:
    return request.app.state.worker


ConfigDep = Annotated[ApiConfig, Depends(get_config)]
WorkerDep = Annotated[RunWorker, Depends(get_worker)]


def sanitize_filename(name: str) -> str:
    """A safe file name: no directories, ASCII letters/digits/._- only, extension kept, stem at most 80 chars."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    stem, dot, extension = base.rpartition(".")
    if not dot:
        stem, extension = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:MAX_FILENAME_STEM] or "dataset"
    extension = re.sub(r"[^A-Za-z0-9]+", "", extension).lower()
    return f"{stem}.{extension}" if extension else stem


def _size_text(n_bytes: int) -> str:
    return f"{n_bytes / 2**20:g} MB" if n_bytes >= 2**20 else f"{n_bytes:,} bytes"


def _save_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    size = 0
    with destination.open("wb") as out:
        while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                raise ApiError(413, "file_too_large", f"The file is larger than the {_size_text(limit)} limit.")
            out.write(chunk)
    if size == 0:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")


def _dataset(dataset_id: str) -> DatasetRecord:
    record = get_repository().get_dataset(dataset_id)
    if record is None or not Path(record.path).is_file():
        raise ApiError(404, "not_found", f"Dataset '{dataset_id}' was not found.")
    return record


def _dataset_profile(record: DatasetRecord) -> DatasetProfile:
    """The profile saved at upload time (recomputed if the cache file is gone)."""
    cache = Path(record.path + PROFILE_SUFFIX)
    if cache.is_file():
        return DatasetProfile.model_validate_json(cache.read_text(encoding="utf-8"))
    profile = profile_dataset(load_dataset(record.path))
    cache.write_text(profile.model_dump_json(), encoding="utf-8")
    return profile


def _context(run_id: str) -> RunContext:
    try:
        return load_context(run_id)
    except (FileNotFoundError, ValueError):  # ValueError: not a valid run id (e.g. path tricks)
        raise ApiError(404, "not_found", f"Run '{run_id}' was not found.") from None


def _require[T](value: T | None, what: str, ctx: RunContext) -> T:
    if value is None:
        raise ApiError(409, "not_ready", f"Run {ctx.run_id} has no {what} yet (state {ctx.state}).")
    return value


def _not_active(run_id: str, worker: RunWorker) -> None:
    if worker.active_run == run_id:
        raise ApiError(409, "run_in_progress", f"Run {run_id} is still running; try again when it has finished.")


def _summary(record: RunRecord, worker: RunWorker) -> RunSummary:
    base = f"/runs/{record.run_id}"
    return RunSummary(
        **record.model_dump(exclude={"report_pdf", "report_md"}),
        progress=STATE_PROGRESS.get(record.state, 0), active=worker.active_run == record.run_id,
        report_pdf_url=f"{base}/report.pdf" if record.report_pdf else None,
        report_md_url=f"{base}/report.md" if record.report_md else None,
    )


def _drive(orchestrator: Orchestrator, run_id: str) -> None:
    """Background job: advance the run until it is DONE, FAILED or waiting for input."""
    try:
        orchestrator.run_until_blocked(run_id)
    except Exception as exc:  # e.g. state.json could not be written: never leave the run looking alive
        logger.exception("Run {} crashed outside its state handlers.", run_id)
        orchestrator.abort(run_id, f"Unexpected error: {type(exc).__name__}: {exc}")


def _report_file(ctx: RunContext, kind: str) -> Path:
    path_text = ctx.report_paths.get(kind)
    if not path_text:
        raise ApiError(409, "not_ready", f"Run {ctx.run_id} has no report yet (state {ctx.state}).")
    path = Path(path_text).resolve()
    if not path.is_relative_to(run_dir(ctx.run_id, create=False).resolve()) or not path.is_file():
        raise ApiError(404, "not_found", f"The report file of run {ctx.run_id} is missing.")
    return path


def _orchestrator(config: ApiConfig, llm: LLMClient | None) -> Orchestrator:
    if config.public_demo:
        return Orchestrator(llm, budget=config.budget, fast_only=True, max_explain_rows=PUBLIC_DEMO_MAX_EXPLAIN_ROWS)
    return Orchestrator(llm, budget=config.budget)


def _abort_interrupted_runs() -> None:
    """Runs left mid-pipeline by a previous server process can never finish: mark them FAILED."""
    unfinished = [s.value for s in RunState if s not in BLOCKED_STATES]
    for record in get_repository().list_runs(limit=LIST_LIMIT, states=unfinished):
        try:
            Orchestrator(None).abort(record.run_id, RESTART_REASON)
            logger.warning("Run {} was interrupted by a restart; marked FAILED.", record.run_id)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Could not mark interrupted run {} as failed: {}", record.run_id, exc)


# --------------------------------------------------------------------------- routes

router = APIRouter(responses={code: {"model": ErrorResponse} for code in (400, 404, 409, 422)})


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(config: ConfigDep) -> HealthResponse:
    """Liveness check. Reports whether an LLM is configured (a boolean only, never the key), whether this is the
    public demo, and the upload limit."""
    return HealthResponse(llm_configured=get_settings().llm_configured, version=API_VERSION,
                          public_demo=config.public_demo, max_upload_mb=config.max_upload_bytes / 2**20)


@router.post("/datasets", response_model=DatasetUploaded, status_code=201, tags=["datasets"],
             responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}})
def upload_dataset(config: ConfigDep, file: Annotated[UploadFile, File(description="CSV, XLSX, JSON or Parquet")]) -> DatasetUploaded:
    """Upload a dataset (max 50 MB; 10 MB in the public demo); returns its id and a profile summary."""
    filename = sanitize_filename(file.filename or "")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ApiError(415, "unsupported_file_type",
                       f"'{extension or filename}' files are not supported. Upload one of: {', '.join(SUPPORTED_EXTENSIONS)}.")
    uploads = get_settings().uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid.uuid4().hex
    path = uploads / f"{dataset_id}_{filename}"
    try:
        _save_upload(file, path, config.max_upload_bytes)
        profile = profile_dataset(load_dataset(path))
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc, ApiError):
            raise
        if isinstance(exc, DataLoadError):
            raise ApiError(400, "invalid_dataset", str(exc).replace(path.name, filename)) from None
        logger.exception("Profiling the upload '{}' failed.", filename)
        raise ApiError(400, "invalid_dataset", f"Could not analyse the file: {type(exc).__name__}: {exc}") from None
    Path(str(path) + PROFILE_SUFFIX).write_text(profile.model_dump_json(), encoding="utf-8")
    record = get_repository().add_dataset(dataset_id, filename, path, profile.n_rows, profile.n_cols)
    logger.info("Uploaded dataset {} ('{}', {:,} x {}).", dataset_id, filename, profile.n_rows, profile.n_cols)
    return DatasetUploaded(**record.model_dump(exclude={"path"}), profile=ProfileSummary.from_profile(profile))


@router.get("/datasets", response_model=list[DatasetOut], tags=["datasets"])
def list_datasets(limit: Annotated[int, Query(ge=1, le=LIST_LIMIT)] = 100,
                  offset: Annotated[int, Query(ge=0)] = 0) -> list[DatasetOut]:
    return [DatasetOut(**r.model_dump(exclude={"path"})) for r in get_repository().list_datasets(limit, offset)]


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfile, tags=["datasets"])
def dataset_profile(dataset_id: str) -> DatasetProfile:
    """The full column-by-column profile of an uploaded dataset."""
    return _dataset_profile(_dataset(dataset_id))


@router.post("/runs", response_model=RunCreated, status_code=202, tags=["runs"],
             responses={403: {"model": ErrorResponse}})
def create_run(body: RunCreate, config: ConfigDep, worker: WorkerDep) -> RunCreated:
    """Start a run on an uploaded dataset. Returns at once; poll GET /runs/{run_id} for progress.
    Only one run can train at a time (409 while busy); the public demo accepts mode "fast" only (403)."""
    if config.public_demo and body.mode != "fast":
        raise ApiError(403, "public_demo", "The public demo runs in fast mode only.")
    dataset = _dataset(body.dataset_id)

    def prepare() -> tuple[str, Callable[[], None]]:
        llm = config.llm_factory()
        orchestrator = _orchestrator(config, llm)
        # Without an LLM the question cannot be parsed: use the profile's suggested task and target instead.
        requirement = None if llm is not None else requirement_from_profile(body.question, _dataset_profile(dataset))
        run_id = orchestrator.start(body.question, dataset.path, requirement=requirement, dataset_id=dataset.id,
                                    actionable_features=body.actionable_features, budget_mode=body.mode)
        return run_id, lambda: _drive(orchestrator, run_id)

    try:
        run_id = worker.submit(prepare)
    except WorkerBusy as exc:
        raise ApiError(409, "busy", f"Run {exc.active_run} is still training and only one run can train at a time. "
                                    "Try again when it has finished.") from None
    return RunCreated(run_id=run_id, state=RunState.INIT, status_url=f"/runs/{run_id}")


@router.get("/runs", response_model=list[RunSummary], tags=["runs"])
def list_runs(worker: WorkerDep, limit: Annotated[int, Query(ge=1, le=LIST_LIMIT)] = 100,
              offset: Annotated[int, Query(ge=0)] = 0, dataset_id: str | None = None,
              state: RunState | None = None) -> list[RunSummary]:
    """Runs, newest first."""
    records = get_repository().list_runs(limit, offset, dataset_id=dataset_id,
                                         states=[state.value] if state is not None else None)
    return [_summary(record, worker) for record in records]


@router.get("/runs/{run_id}", response_model=RunDetail, tags=["runs"])
def get_run(run_id: str, worker: WorkerDep) -> RunDetail:
    """State, progress %, events, metrics, verification and any questions waiting for an answer."""
    ctx = _context(run_id)
    training = ctx.training
    summary = _summary(run_record(ctx), worker)
    message = ctx.message
    for path, url in ((ctx.report_paths.get("pdf"), summary.report_pdf_url), (ctx.report_paths.get("markdown"), summary.report_md_url)):
        if message and path and url:
            message = message.replace(path, url)  # the API shows report URLs, never server paths
    return RunDetail(
        **summary.model_dump(),
        message=message,
        target=training.target if training else (ctx.requirement.target if ctx.requirement else None),
        test_metrics=training.test_metrics if training else {},
        decision_threshold=training.decision_threshold if training else None,
        verification=ctx.verification, pending_questions=ctx.pending_questions, answers=ctx.answers,
        errors=ctx.errors,
        events=[RunEventOut(ts=e.timestamp, from_state=e.from_state, to_state=e.state, message=e.message)
                for e in ctx.history],
    )


@router.post("/runs/{run_id}/answer", response_model=RunCreated, status_code=202, tags=["runs"])
def answer_run(run_id: str, body: AnswerRequest, config: ConfigDep, worker: WorkerDep) -> RunCreated:
    """Answer the clarifying questions of a run in NEEDS_INPUT; the run then continues in the background."""
    ctx = _context(run_id)
    if ctx.state != RunState.NEEDS_INPUT:
        raise ApiError(409, "invalid_state", f"Run {run_id} is in state {ctx.state}, not waiting for an answer.")

    def prepare() -> tuple[str, Callable[[], None]]:
        orchestrator = _orchestrator(config, config.llm_factory())
        with worker.run_lock(run_id):
            try:
                orchestrator.provide_answer(run_id, body.answer)
            except ValueError as exc:  # e.g. answered twice concurrently
                raise ApiError(409, "invalid_state", str(exc)) from None
        return run_id, lambda: _drive(orchestrator, run_id)

    try:
        worker.submit(prepare)
    except WorkerBusy as exc:
        raise ApiError(409, "busy", f"Run {exc.active_run} is still training; try again when it has finished.") from None
    return RunCreated(run_id=run_id, state=RunState.PARSE, status_url=f"/runs/{run_id}")


@router.get("/runs/{run_id}/leaderboard", response_model=LeaderboardResponse, tags=["results"])
def leaderboard(run_id: str) -> LeaderboardResponse:
    ctx = _context(run_id)
    training = _require(ctx.training, "trained models", ctx)
    return LeaderboardResponse(run_id=run_id, primary_metric=training.primary_metric, direction=training.direction,
                               best_model=training.best_model, entries=training.leaderboard)


@router.get("/runs/{run_id}/explanations", response_model=ExplanationResponse, tags=["results"])
def explanations(run_id: str) -> ExplanationResponse:
    """Key drivers (SHAP or permutation importance), local examples, clusters or forecast, and chart URLs."""
    ctx = _context(run_id)
    explanation = _require(ctx.explanation, "explanations", ctx)
    charts = {name: f"/runs/{run_id}/charts/{Path(path).name}" for name, path in explanation.charts.items()}
    return ExplanationResponse(run_id=run_id, **explanation.model_dump(exclude={"charts"}), charts=charts)


@router.get("/runs/{run_id}/recommendations", response_model=RecommendationsResponse, tags=["results"])
def recommendations(run_id: str) -> RecommendationsResponse:
    ctx = _context(run_id)
    result = _require(ctx.recommendations, "recommendations", ctx)
    return RecommendationsResponse(run_id=run_id, **result.model_dump(), what_if_example=ctx.what_if_example)


@router.get("/runs/{run_id}/insight", response_model=DecisionInsight, tags=["results"])
def insight(run_id: str) -> DecisionInsight:
    """The plain-language decision brief."""
    ctx = _context(run_id)
    return _require(ctx.insight, "decision brief", ctx)


@router.post("/runs/{run_id}/what-if", response_model=WhatIfResult | SegmentImpact, tags=["results"])
def what_if(run_id: str, body: WhatIfRequest, worker: WorkerDep) -> WhatIfResult | SegmentImpact:
    """Simulate changes with the run's trained model, for one record (base_row) or a filtered segment."""
    _not_active(run_id, worker)
    with worker.run_lock(run_id):
        ctx = _context(run_id)
        _require(ctx.training, "trained model", ctx)
        try:
            return Orchestrator(None).what_if(run_id, body.changes, base_row=body.base_row, segment=body.segment)
        except (WhatIfError, ValueError) as exc:
            raise ApiError(400, "invalid_what_if", str(exc)) from None


@router.post("/runs/{run_id}/ask", response_model=AnalysisAnswer, tags=["results"],
             responses={403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def ask(run_id: str, body: AskRequest, config: ConfigDep, worker: WorkerDep) -> AnalysisAnswer:
    """Answer a question about the run's data with LLM-written pandas code run in the sandbox.
    Disabled in the public demo (403), because it executes generated code on the server."""
    if config.public_demo:
        raise ApiError(403, "public_demo", "Ask Your Data is disabled in the public demo (it runs generated code).")
    _not_active(run_id, worker)
    _context(run_id)
    llm = config.llm_factory()
    if llm is None:
        raise ApiError(503, "llm_not_configured", "Asking questions needs an LLM; set LLM_MODEL and LLM_API_KEY in .env.")
    with worker.run_lock(run_id):
        return Orchestrator(llm).ask(run_id, body.question)


@router.get("/runs/{run_id}/report.pdf", response_class=FileResponse, tags=["reports"])
def report_pdf(run_id: str) -> FileResponse:
    path = _report_file(_context(run_id), "pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"idss_report_{run_id}.pdf")


@router.get("/runs/{run_id}/report.md", response_class=FileResponse, tags=["reports"])
def report_markdown(run_id: str) -> FileResponse:
    path = _report_file(_context(run_id), "markdown")
    return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=f"idss_report_{run_id}.md")


@router.get("/runs/{run_id}/charts/{name}", response_class=FileResponse, tags=["reports"])
def chart(run_id: str, name: str) -> FileResponse:
    """A chart PNG of the run, by file name (e.g. feature_importance.png). Only files in the run's charts folder."""
    if not _CHART_NAME.fullmatch(name):
        raise ApiError(400, "bad_request", "Invalid chart name.")
    try:
        charts_dir = (run_dir(run_id, create=False) / "charts").resolve()
    except ValueError:
        raise ApiError(404, "not_found", f"Run '{run_id}' was not found.") from None
    path = (charts_dir / (name if name.endswith(".png") else f"{name}.png")).resolve()
    if path.parent != charts_dir or not path.is_file():
        raise ApiError(404, "not_found", f"Chart '{name}' was not found for run {run_id}.")
    return FileResponse(path, media_type="image/png")


# --------------------------------------------------------------------------- errors and middleware


def _error(status_code: int, error: str, detail: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponse(error=error, detail=detail).model_dump(mode="json"))


def _error_code(status_code: int) -> str:
    return _ERROR_CODES.get(status_code) or HTTPStatus(status_code).phrase.lower().replace(" ", "_")


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return _error(exc.status_code, exc.error, exc.detail)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return _error(exc.status_code, _error_code(exc.status_code), exc.detail)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
                for e in exc.errors()]
    return _error(422, "validation_error", problems)


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on {} {}.", request.method, request.url.path)
    return _error(500, "internal_error", "Unexpected server error; see the server log.")


class UploadSizeLimit:
    """Reject an upload whose declared Content-Length is over the limit before its body is read."""

    def __init__(self, app: ASGIApp, config: ApiConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"].rstrip("/") == "/datasets":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > self.config.max_upload_bytes + MULTIPART_OVERHEAD_BYTES:
                response = _error(413, "file_too_large",
                                  f"The file is larger than the {_size_text(self.config.max_upload_bytes)} limit.")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- app


def create_app(*, llm_factory: Callable[[], LLMClient | None] | None = None, budget: TrainingBudget | None = None,
               max_upload_bytes: int = MAX_UPLOAD_BYTES, public_demo: bool | None = None) -> FastAPI:
    """Build the API. Tests inject a fake LLM factory, a tiny training budget or a small upload limit.
    `public_demo` defaults to the IDSS_PUBLIC_DEMO setting; it caps uploads at 10 MB."""
    demo = get_settings().public_demo if public_demo is None else public_demo
    if demo:
        max_upload_bytes = min(max_upload_bytes, PUBLIC_DEMO_MAX_UPLOAD_BYTES)
    config = ApiConfig(llm_factory=llm_factory or default_llm_factory, budget=budget, max_upload_bytes=max_upload_bytes,
                       public_demo=demo)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        ensure_workspace()  # runs/, logs/, uploads/ (a fresh clone or an ephemeral cloud disk may lack them)
        init_db()
        if config.public_demo:
            purge_older_than(PUBLIC_DEMO_RETENTION)
        _abort_interrupted_runs()
        yield

    app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan,
                  description="Intelligent Decision Support System: upload data, ask a business question, "
                              "and get a verified model, explanations, recommendations and a report.")
    app.state.config = config
    app.state.worker = RunWorker()
    app.include_router(router)
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)
    app.add_middleware(UploadSizeLimit, config=config)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
    return app


app = create_app()
