"""Safe runner for LLM-generated pandas analysis snippets.

Defence in depth:
1. Static guard (ast) before anything runs: only allow-listed imports; no file/OS/network/introspection
   builtins; no private or dunder attributes; no pandas/numpy file I/O or string-evaluation methods.
2. Separate process: `python -I` (isolated mode) running sandbox_runner.py with restricted builtins and a
   guarded __import__, in a fresh temporary directory, with an environment stripped of secrets.
3. Limits: wall-clock timeout (the process is killed), stdout capped at 20 KB, result rows/chars capped.

Not a perfect security boundary (no memory limit on Windows); it is meant for short analysis snippets.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from idss.core.logger import get_logger
from idss.ml.data_loader import load_dataset

logger = get_logger()

RUNNER = Path(__file__).resolve().parent / "sandbox_runner.py"
ALLOWED_MODULES = frozenset({"pandas", "numpy", "math", "statistics", "datetime", "json", "re"})
BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "input", "breakpoint", "globals", "locals", "vars", "getattr", "setattr", "delattr", "help", "exit", "quit",
    "memoryview", "importlib", "builtins",
})
BLOCKED_ATTRIBUTES = frozenset({
    # string evaluation
    "eval", "query",
    # numpy file I/O and native-code escape hatches
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "open_memmap", "fromregex", "lib", "ctypeslib", "f2py", "distutils",
    # pandas/numpy internals and I/O namespaces and writers (read_*/to_* are matched by _IO_ATTRIBUTE)
    "io", "core", "ExcelWriter", "ExcelFile", "HDFStore", "DataSource",
    # process control, in case a module object leaks in
    "system", "popen",
})
_IO_ATTRIBUTE = re.compile(
    r"^(read_\w+|to_(csv|excel|pickle|parquet|feather|hdf|sql|stata|clipboard|orc|xml|html|latex|json|markdown|gbq))$"
)
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "int",
    "isinstance", "issubclass", "len", "list", "map", "max", "min", "next", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip", "Exception", "ValueError", "KeyError",
    "TypeError", "IndexError", "ZeroDivisionError", "StopIteration", "ArithmeticError", "AttributeError",
)
_ENV_ALLOWLIST = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
                  "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
MAX_CODE_CHARS = 20_000
MAX_STDOUT_CHARS = 20 * 1024
MAX_RESULT_ROWS = 200
MAX_RESULT_CHARS = 50_000
DEFAULT_TIMEOUT_S = 20.0


class SandboxResult(BaseModel):
    ok: bool
    stdout: str = ""
    result: Any = None
    error: str | None = None
    duration_s: float = 0.0
    truncated: bool = False


def check_code(code: str) -> list[str]:
    """Reasons the code is not allowed to run (empty list = allowed)."""
    if len(code) > MAX_CODE_CHARS:
        return [f"code is longer than {MAX_CODE_CHARS} characters"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [f"import of '{a.name}' is not allowed" for a in node.names if not module_allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not module_allowed(module):
                problems.append(f"import from '{module or '.'}' is not allowed")
            problems += [f"importing '{a.name}' is not allowed" for a in node.names if _blocked_member(a.name)]
        elif isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            problems.append(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and _blocked_member(node.attr):
            problems.append(f"attribute '.{node.attr}' is not allowed")
    return list(dict.fromkeys(problems))


def module_allowed(name: str) -> bool:
    """An allow-listed top-level module, without private or blocked submodules (e.g. pandas.io, numpy.lib)."""
    root, *rest = name.split(".")
    return root in ALLOWED_MODULES and not any(_blocked_member(part) for part in rest)


def _blocked_member(name: str) -> bool:
    return name.startswith("_") or name in BLOCKED_ATTRIBUTES or bool(_IO_ATTRIBUTE.match(name))


def run_code(code: str, df_path: str | Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Run `code` against the dataset at `df_path` (available as `df`); never raises."""
    start = time.perf_counter()

    def elapsed() -> float:
        return round(time.perf_counter() - start, 3)

    problems = check_code(code)
    if problems:
        return SandboxResult(ok=False, error="Blocked by the sandbox: " + "; ".join(problems), duration_s=elapsed())

    with tempfile.TemporaryDirectory(prefix="idss-sandbox-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        try:
            load_dataset(df_path).to_parquet(workdir / "data.parquet")
        except Exception as exc:
            return SandboxResult(ok=False, error=f"Could not load the data: {exc}", duration_s=elapsed())
        config = {
            "code": code,
            "data_path": str(workdir / "data.parquet"),
            "output_path": str(workdir / "output.json"),
            "allowed_modules": sorted(ALLOWED_MODULES),
            "blocked_members": sorted(BLOCKED_ATTRIBUTES),
            "io_member_pattern": _IO_ATTRIBUTE.pattern,
            "safe_builtins": list(SAFE_BUILTINS),
            "max_stdout_chars": MAX_STDOUT_CHARS,
            "max_result_rows": MAX_RESULT_ROWS,
            "max_result_chars": MAX_RESULT_CHARS,
        }
        (workdir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(workdir / "config.json")],
                cwd=workdir, env=_child_env(), capture_output=True, timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Sandbox code timed out after {}s and was stopped.", timeout_s)
            return SandboxResult(ok=False, error=f"Timed out after {timeout_s:g}s; the code was stopped.",
                                 duration_s=elapsed())

        output = workdir / "output.json"
        if not output.is_file():
            stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            return SandboxResult(ok=False, error=f"Sandbox process failed (exit code {completed.returncode}): {stderr}",
                                 duration_s=elapsed())
        data = json.loads(output.read_text(encoding="utf-8"))
    return SandboxResult(ok=data["ok"], stdout=data["stdout"], result=data["result"], error=data["error"],
                         truncated=data["truncated"], duration_s=elapsed())


def _child_env() -> dict[str, str]:
    """Only what Python needs to start; API keys and other secrets are never passed to the child."""
    env = {name: value for name, value in os.environ.items() if name.upper() in _ENV_ALLOWLIST}
    env["PYTHONIOENCODING"] = "utf-8"
    return env
