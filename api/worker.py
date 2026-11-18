"""Background execution of runs: one training job at a time, on its own thread.

A dedicated daemon thread (rather than the request's thread pool) keeps POST /runs non-blocking and lets
long training jobs outlive the request. Only one job may be active at a time so a laptop stays responsive;
a second submission raises WorkerBusy. Per-run locks serialise everything that rewrites a run's state.json
(the job itself, answers, what-if and ask), so concurrent requests never overwrite each other's changes.
"""

import threading
from collections import defaultdict
from collections.abc import Callable

from idss.core.logger import get_logger

logger = get_logger()

_STARTING = "<starting>"


class WorkerBusy(Exception):
    def __init__(self, active_run: str) -> None:
        super().__init__(f"Run {active_run} is still running.")
        self.active_run = active_run


class RunWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: str | None = None
        self._thread: threading.Thread | None = None
        self._run_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    @property
    def active_run(self) -> str | None:
        """Id of the run being executed (None when idle)."""
        active = self._active
        return None if active == _STARTING else active

    @property
    def busy(self) -> bool:
        return self._active is not None

    def run_lock(self, run_id: str) -> threading.Lock:
        """The lock that guards writes to one run's state.json."""
        with self._lock:
            return self._run_locks[run_id]

    def submit(self, prepare: Callable[[], tuple[str, Callable[[], object]]]) -> str:
        """Reserve the worker, call `prepare()` -> (run_id, job) in the caller's thread, then run `job` in the
        background. Raises WorkerBusy if a job is active; if `prepare` raises, the reservation is released."""
        with self._lock:
            if self._active is not None:
                raise WorkerBusy(self.active_run or "another run")
            self._active = _STARTING
        try:
            run_id, job = prepare()
        except BaseException:
            with self._lock:
                self._active = None
            raise
        thread = threading.Thread(target=self._execute, args=(run_id, job), name=f"idss-run-{run_id}", daemon=True)
        with self._lock:
            self._active, self._thread = run_id, thread
        thread.start()
        return run_id

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the current job finishes; True if the worker is idle afterwards."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.busy

    def _execute(self, run_id: str, job: Callable[[], object]) -> None:
        try:
            with self.run_lock(run_id):
                job()
        except Exception:  # the orchestrator already turns handler errors into FAILED; this is a last resort
            logger.exception("Background job for run {} crashed.", run_id)
        finally:
            with self._lock:
                self._active = None
            logger.info("Worker finished run {}.", run_id)
